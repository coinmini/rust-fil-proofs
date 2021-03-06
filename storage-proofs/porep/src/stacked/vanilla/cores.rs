use log::*;
use std::sync::{Mutex, MutexGuard};

use anyhow::Result;
use hwloc::{ObjectType, Topology, TopologyObject, CPUBIND_THREAD};
use lazy_static::lazy_static;

use storage_proofs_core::settings;

type CoreGroup = Vec<CoreIndex>;

//第一步：定义core_groups， 获取 Cores: 32, Shared Caches: 8, cores per cache (group_size): 4
//这个是第一调用的方法，core_groups(cores_per_unit)

lazy_static! {
    pub static ref TOPOLOGY: Mutex<Topology> = Mutex::new(Topology::new());
    pub static ref CORE_GROUPS: Option<Vec<Mutex<CoreGroup>>> = {

        //获取环境变量定义的multicore_sdr_producers 的数量
        let settings = &settings::SETTINGS;
        //默认multicore_sdr_producers = 3 ，cores_per_unit = 3 + 1 =4
        let num_producers = settings.multicore_sdr_producers;
        let cores_per_unit = num_producers + 1;

        core_groups(cores_per_unit)
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// `CoreIndex` is a simple wrapper type for indexes into the set of vixible cores. A `CoreIndex` should only ever be
/// created with a value known to be less than the number of visible cores.
pub struct CoreIndex(usize);


//第二步： 检查并锁定 core group 0 locked, could not checkout
//这里锁group，如果1个group被用了，这个group 就被锁了。所以关键在定义group
// core group 编号从0开始到7
// checked out core group 0 或者是 core group 0 locked, could not checkout"，去找下一个group
pub fn checkout_core_group() -> Option<MutexGuard<'static, CoreGroup>> {
    match &*CORE_GROUPS {
        Some(groups) => {
            for (i, group) in groups.iter().enumerate() {
                match group.try_lock() {
                    Ok(guard) => {
                        debug!("checked out core group {}", i);
                        return Some(guard);
                    }
                    Err(_) => debug!("core group {} locked, could not checkout", i),
                }
            }
            None
        }
        None => None,
    }
}

#[cfg(not(target_os = "windows"))]
pub type ThreadId = libc::pthread_t;

#[cfg(target_os = "windows")]
pub type ThreadId = winapi::winnt::HANDLE;

/// Helper method to get the thread id through libc, with current rust stable (1.5.0) its not
/// possible otherwise I think.
#[cfg(not(target_os = "windows"))]
fn get_thread_id() -> ThreadId {
    unsafe { libc::pthread_self() }
}

#[cfg(target_os = "windows")]
fn get_thread_id() -> ThreadId {
    unsafe { kernel32::GetCurrentThread() }
}

pub struct Cleanup {
    tid: ThreadId,
    prior_state: Option<hwloc::Bitmap>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        match self.prior_state.take() {
            Some(prior) => {
                let child_topo = &TOPOLOGY;
                let mut locked_topo = child_topo.lock().unwrap();
                let _ = locked_topo.set_cpubind_for_thread(self.tid, prior, CPUBIND_THREAD);
            }
            None => (),
        }
    }
}
//第三步，绑定 consumer 到main thread 核心，4个线程中的第一个；
//第四步，绑定producer的核心
// allowed cpuset: 0,32
//bind_core 方法给 multi.rs里面用
pub fn bind_core(core_index: CoreIndex) -> Result<Cleanup> {
    let child_topo = &TOPOLOGY;
    let tid = get_thread_id();
    let mut locked_topo = child_topo.lock().unwrap();
    let core = get_core_by_index(&locked_topo, core_index).map_err(|err| {
        anyhow::format_err!("failed to get core at index {}: {:?}", core_index.0, err)
    })?;

    let cpuset = core.allowed_cpuset().ok_or_else(|| {
        anyhow::format_err!("no allowed cpuset for core at index {}", core_index.0,)
    })?;
    debug!("allowed cpuset: {:?}", cpuset);
    let mut bind_to = cpuset;

    // Get only one logical processor (in case the core is SMT/hyper-threaded).
    bind_to.singlify();

    // Thread binding before explicit set.
    let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);

    debug!("binding to {:?}", bind_to);
    // Set the binding.
    let result = locked_topo
        .set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD)
        .map_err(|err| anyhow::format_err!("failed to bind CPU: {:?}", err));

    if result.is_err() {
        warn!("error in bind_core, {:?}", result);
    }

    Ok(Cleanup {
        tid,
        prior_state: before,
    })
}

fn get_core_by_index<'a>(topo: &'a Topology, index: CoreIndex) -> Result<&'a TopologyObject> {
    let idx = index.0;

    match topo.objects_with_type(&ObjectType::Core) {
        Ok(all_cores) if idx < all_cores.len() => Ok(all_cores[idx]),
        Ok(all_cores) => Err(anyhow::format_err!(
            "idx ({}) out of range for {} cores",
            idx,
            all_cores.len()
        )),
        _e => Err(anyhow::format_err!("failed to get core by index {}", idx,)),
    }
}

fn core_groups(cores_per_unit: usize) -> Option<Vec<Mutex<Vec<CoreIndex>>>> {
    let topo = TOPOLOGY.lock().unwrap();


    let core_depth = match topo.depth_or_below_for_type(&ObjectType::Core) {
        Ok(depth) => depth,
        Err(_) => return None,
    };


    //获取多少个核心，比如3970是32个核心
    let all_cores = topo.objects_with_type(&ObjectType::Core).unwrap();
    let core_count = all_cores.len();

    let mut cache_depth = core_depth;
    let mut cache_count = 0;

    //获取 L3 缓存组cache_count的数量， 8 
    while cache_depth > 0 {
        let objs = topo.objects_at_depth(cache_depth);
        let obj_count = objs.len();
        if obj_count < core_count {
            cache_count = obj_count;
            break;
        }

        cache_depth -= 1;
    }

    assert_eq!(0, core_count % cache_count);
    //group_size = 32 / 8 /2 = 2 
    // group_count =  2 * 8  = 16
    let mut group_size = (core_count / cache_count) / 2 ;
    let mut group_count = 2 * cache_count;


    if cache_count <= 1 {
        // If there are not more than one shared caches, there is no benefit in trying to group cores by cache.
        // In that case, prefer more groups so we can still bind cores and also get some parallelism.
        // Create as many full groups as possible. The last group may not be full.

       
        group_count = core_count / cores_per_unit; // 32 / 4  = 8 
        group_size = cores_per_unit;  // 4 

        info!(
            "found only {} shared cache(s), heuristically grouping cores into {} groups",
            cache_count, group_count
        );
    } else {
        debug!(

            // 3970 是设置 cores_per_unit = 4 ， 就是3+1个线程跑1个任务； 32 / 4 = 8 
            // 把group_size 设置为2 ， cache_count 设置为 16

            // Cores: 32, Shared Caches: 8, cores per cache (group_size): 4
            "Cores: {}, Shared Caches: {}, cores per cache (group_size): {}",
            core_count, cache_count, group_size
        );
    }

    // 定义好core_groups ，是 0到group_count 到总数
    // 这里就是绑定核心到真正地方了，core_index

    //(1..20).map(|x| x+1); 是指迭代器的所有元素 自加1
    // （0..8）.map(|i|   {0..4}.map(|j| {实际运算}))，  类似for循环嵌套
    let core_groups = (0..group_count)
        .map(|i| {
            (0..group_size)
                .map(|j| {
                    //获取线程核心序号进行绑定
                    let core_index = i * group_size + j;
                    assert!(core_index < core_count);

                    // 这个CoreInde给上面到bind_core 方法里面用
                    CoreIndex(core_index)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Some(
        core_groups
            .iter()
            .map(|group| Mutex::new(group.clone()))
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cores() {
        core_groups(2);
    }

    #[test]
    fn test_checkout_cores() {
        let checkout1 = checkout_core_group();
        dbg!(&checkout1);
        let checkout2 = checkout_core_group();
        dbg!(&checkout2);

        // This test might fail if run on a machine with fewer than four cores.
        match (checkout1, checkout2) {
            (Some(c1), Some(c2)) => assert!(*c1 != *c2),
            _ => panic!("failed to get two checkouts"),
        }
    }
}
