---
icon: simple/speedtest
---

# Tips for Low Level Optimizations on AMDGPU (ROCm)

The ROCm platform allows for device code to be compiled to the final ISA called
~GCN~. The compilation goes through the amdgpu backend in llvm, using a
number of llvm binutil tools like the assembler or linker, and produces Elf
object files.

This guide focuses on some manual workflows enabled by this layering and by
developer options built into the IREE tooling.


## Hand-modifying GCN assembly

Sometimes it is much faster to explore ISA-level optimizations by modifying
compiler output instead of changing the compiler.

### Dumping and Inspecting Device Code

To get started, add `--iree-hal-dump-executable-files-to=dump` to your compile
command. When targetting hip/ROCm, this will result in the following files
saved to the `dump` directory:

```
module_main_0_dispatch_0.mlir
module_main_0_dispatch_0_rocm_hsaco_fb_benchmark.mlir
module_main_0_dispatch_0_rocm_hsaco_fb.hsaco
module_main_0_dispatch_0_rocm_hsaco_fb.linked.ll
module_main_0_dispatch_0_rocm_hsaco_fb.optimized.ll
module_main_0_dispatch_0_rocm_hsaco_fb.rocmasm
```

(Or similar for other dispatches.)

The `.hsaco` file is a ROCm shared object with device code in the HSA Code
Object format:

```shell
$ export PATH="<iree-build-dir>/llvm-project/bin:$PATH"
$ cd dump
$ llvm-readelf --file-header module_main_0_dispatch_0_rocm_hsaco_fb.hsaco
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 40 03 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            40
  ABI Version:                       3
  Type:                              DYN (Shared object file)
  Machine:                           EM_AMDGPU
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          64 (bytes into file)
  Start of section headers:          6024 (bytes into file)
  Flags:                             0xE4C
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         8
  Size of section headers:           64 (bytes)
  Number of section headers:         14
  Section header string table index: 12
```

While the `.rocasm` file contains the device code in a disassembled form:

```shell
$ cat module_main_0_dispatch_0_rocm_hsaco_fb.rocmasm | head -n 10
        .text
        .amdgcn_target "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"
        .globl  main_0_dispatch_0_contract_2048x10240x1280_f16xf16xf32
        .p2align        8
        .type   main_0_dispatch_0_contract_2048x10240x1280_f16xf16xf32,@function
main_0_dispatch_0_contract_2048x10240x1280_f16xf16xf32:
        s_trap 2 ; Trap with incompatible firmware that doesn't support preloading kernel arguments.
        s_nop 0
        s_nop 0
        s_nop 0
        ...
```

Worth highlighting, the end of the disassembly contains a metadata section:

```shell
$ cat module_main_0_dispatch_0_rocm_hsaco_fb.rocmasm | tail -n 40
.section        ".note.GNU-stack","",@progbits
        .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 256
    .name:           main_0_dispatch_0_contract_2048x10240x1280_f16xf16xf32
    .private_segment_fixed_size: 0
    .sgpr_count:     19
    .sgpr_spill_count: 0
    .symbol:         main_0_dispatch_0_contract_2048x10240x1280_f16xf16xf32.kd
    .uses_dynamic_stack: false
    .vgpr_count:     102
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-'
amdhsa.version:
  - 1
  - 2
...

        .end_amdgpu_metadata
```


### Reassembling Modified Device Code

After you modify a `.rocasm` file, save it to a new directory `replace`:

```shell
$ ls replace
module_main_0_dispatch_0_rocm_hsaco_fb.rocmasm
```

Next, re-assemble it using the `clang` from IREE:

```shell
$ clang -cc1as -filetype obj -triple amdgcn-amd-amdhsa -target-cpu gfx942 \
    -target-feature +sramecc -target-feature -xnack -mrelocation-model pic \
    -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
    -o clang.o module_main_0_dispatch_0_rocm_hsaco_fb.rocmasm
```

This creates a relocatable Elf file:
```shell
$ llvm-readelf --file-header module_main_0_dispatch_0_rocm_hsaco_fb.hsaco
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 40 03 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            40
  ABI Version:                       3
  Type:                              REL (Relocatable file)
  Machine:                           EM_AMDGPU
  Version:                           0x1
  Entry point address:               0x0
  Start of program headers:          0 (bytes into file)
  Start of section headers:          5112 (bytes into file)
  Flags:                             0xE4C
  Size of this header:               64 (bytes)
  Size of program headers:           0 (bytes)
  Number of program headers:         0
  Size of section headers:           64 (bytes)
  Number of section headers:         8
  Section header string table index: 1
```

Next, link it to create a shared object:

```shell
$ lld -flavor gnu -m elf64_amdgpu --no-undefined -shared \
    -plugin-opt=-amdgpu-internalize-symbols -plugin-opt=mcpu=gfx942 \
    --no-whole-archive -o module_main_0_dispatch_0_rocm_hsaco_fb.hsaco clang.o
```

To confirm that the produced shared object file contains your modification, you
disassemble each and compare them:

```shel
$ cd ..
$ /opt/rocm/lib/llvm/bin/llvm-objdump -D --triple=amdgcn-amd-amdhsa --mcpu=gfx942 \
    dump/module_main_0_dispatch_0_rocm_hsaco_fb.hsaco > orig.s
$ /opt/rocm/lib/llvm/bin/llvm-objdump -D --triple=amdgcn-amd-amdhsa --mcpu=gfx942 \
    replace/module_main_0_dispatch_0_rocm_hsaco_fb.hsaco > replace.s
$ vimdiff orig.s replace.s
```

NOTE: The disassembly format used by the command above cannot be used to
reassemble the file again.


```shell
$ bazel run //tools:iree-compile -- \
  --iree-hal-target-backends=vmvx \
  $PWD/samples/models/simple_abs.mlir \
  -o /tmp/module.fb
```

and then benchmark an exported function in that module:

```shell
$ bazel run //tools:iree-benchmark-module -- \
  --module=/tmp/module.fb \
  --device=local-task \
  --function=abs \
  --input=f32=-2
```

You'll see output like

```shell
Run on (12 X 4500 MHz CPU s)
CPU Caches:
  L1 Data 32K (x6)
  L1 Instruction 32K (x6)
  L2 Unified 1024K (x6)
  L3 Unified 8448K (x1)
Load Average: 2.21, 1.93, 3.34
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may
 be noisy and will incur extra overhead.
***WARNING*** Library was built as DEBUG. Timings may be affected.
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_RunModule/process_time/real_time       0.22 ms         0.23 ms         3356
```

Notice that there are a few warnings in there (you may not see all of these).
The benchmark library helpfully warns about some common issues that will affect
benchmark timing. When trying to obtain real benchmark numbers, you should
generally build an optimized build (`-c opt` in Bazel) and
[disable CPU scaling](#cpu-configuration).

```shell
bazel build -c opt //tools:iree-benchmark-module
```

Another thing to consider is that depending on where you are running the
benchmark you might want to avoid additional programs running at the same time.
Bazel itself runs a server even when it's not being actively invoked that can be
quite a memory hog, so we'll instead invoke the binary directly. Use your
favorite process manager (e.g. [htop](https://hisham.hm/htop/) or
[pkill](https://en.wikipedia.org/wiki/Pkill) on Linux) to kill heavy-weight
programs such as Chrome and Bazel.

Now we'll actually invoke the binary:

```shell
$ ./bazel-bin/tools/iree-benchmark-module \
  --module=/tmp/module.fb \
  --device=local-task \
  --function=abs \
  --input=f32=-2
```

```shell
Run on (12 X 4500 MHz CPU s)
CPU Caches:
  L1 Data 32K (x6)
  L1 Instruction 32K (x6)
  L2 Unified 1024K (x6)
  L3 Unified 8448K (x1)
Load Average: 1.49, 3.42, 3.49
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_RunModule/process_time/real_time      0.011 ms        0.014 ms        61654
```

Remember to [restore CPU scaling](#cpu-configuration) when you're done.

## Executable Benchmarks

We also benchmark the performance of individual parts of the IREE system in
isolation. IREE breaks a model down to dispatch functions. To benchmark all the
dispatch functions, generate an IREE module with the
`-iree-flow-export-benchmark-funcs` flag set:

```shell
$ build/tools/iree-compile \
  --iree-input-type=stablehlo \
  --iree-flow-export-benchmark-funcs \
  --iree-hal-target-backends=vmvx \
  tests/e2e/stablehlo_models/fullyconnected.mlir \
  -o /tmp/fullyconnected.vmfb
```

and then benchmark all exported dispatch functions (and all exported functions)
in that module:

```shell
$ build/tools/iree-benchmark-module
  --module=/tmp/fullyconnected.vmfb
  --device=local-task
```

If no `entry_function` is specified, `iree-benchmark-module` will register a
benchmark for each exported function that takes no inputs.

You will see output like:

```shell
Run on (72 X 3700 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x36)
  L1 Instruction 32 KiB (x36)
  L2 Unified 1024 KiB (x36)
  L3 Unified 25344 KiB (x2)
Load Average: 4.39, 5.72, 6.76
---------------------------------------------------------------------------------------------
Benchmark                                                   Time             CPU   Iterations
---------------------------------------------------------------------------------------------
BM_main_ex_dispatch_0_benchmark/process_time/real_time  0.030 ms        0.037 ms        34065
BM_main_ex_dispatch_1_benchmark/process_time/real_time  0.034 ms        0.042 ms        20567
BM_main_ex_dispatch_2_benchmark/process_time/real_time  0.043 ms        0.051 ms        18576
BM_main_ex_dispatch_3_benchmark/process_time/real_time  0.029 ms        0.036 ms        21345
BM_main_ex_dispatch_4_benchmark/process_time/real_time  0.042 ms        0.051 ms        15880
BM_main_ex_dispatch_5_benchmark/process_time/real_time  0.030 ms        0.037 ms        17854
BM_main_ex_dispatch_6_benchmark/process_time/real_time  0.043 ms        0.052 ms        14919
BM_main_benchmark/process_time/real_time                0.099 ms        0.107 ms         5892
```

### Bytecode Module Benchmarks

Normally, the IREE VM is expected to be integrated into applications and driving
model execution. So its performance is of crucial importance. We strive to
introduce as little overhead as possible and have several benchmark binaries
dedicated for evaluating the VM's performance. These benchmark binaries are
named as `*_benchmark` in the
[`iree/vm/`](https://github.com/openxla/iree/tree/main/runtime/src/iree/vm)
directory. They also use the Google Benchmark library as the above.

## CPU Configuration

When benchmarking, it's important to consider the configuration of your CPUs.
Most notably, CPU scaling can give variable results, so you'll usually want to
disable it. This can get pretty complex, but the most basic thing to do is to
run all CPUs at maximum frequency. The other thing to consider is what CPU(s)
your program is running on. Both of these get more complicated on mobile and in
multithreaded workloads.

### Linux

Google benchmark provides some
[instructions](https://github.com/google/benchmark#disabling-cpu-frequency-scaling).
Note that the library will print "CPU scaling is enabled" warnings for any
configuration that
[doesn't have the quota governor set to performance](https://github.com/google/benchmark/blob/3d1c2677686718d906f28c1d4da001c42666e6d2/src/sysinfo.cc#L228).
Similarly the CPU frequency it reports is the
[maximum frequency of cpu0](https://github.com/google/benchmark/blob/3d1c2677686718d906f28c1d4da001c42666e6d2/src/sysinfo.cc#L533),
not the frequency of the processor it's actually running on. This means that
more advanced configurations should ignore these messages.

Turn off CPU scaling before benchmarking.

```shell
sudo cpupower frequency-set --governor performance
```

Restore CPU scaling after benchmarking:

```shell
sudo cpupower frequency-set --governor powersave
```

To learn more about different quota
governor settings, see
<https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt>. To restrict
which CPUs you run on, use the `taskset` command which takes a hexadecimal mask.

To only run on the lowest-numbered CPU you can run

```shell
taskset 1 sleep 20 &
```

You can confirm that the process is running on the given CPU:

```shell
ps -o psr $!
```

Note that `$!` indicates the process ID of the last executed background command,
so you can only use this shorthand if you didn't run any commands after the
sleep. For more info on taskset, see <https://linux.die.net/man/1/taskset>.

### Android

Read and understand the [Linux](#linux) instructions first.

Android doesn't give us quite as nice tooling, but the principle is basically
the same. One important difference is that thermal throttling is a much bigger
concern on mobile. Without a cooling plate, it is likely that high clock speeds
will overheat the device and engage thermal throttling, which will ignore
whatever clock speeds you may have set to prevent things from catching on fire.
Therefore the naive approach above is likely not a good idea.

You will likely need to be root (use `su` or `adb root`). The commands will
depend on your exact phone and number of cores. First play around and make sure
you understand what everything means. Note that each CPU has its own files which
are used to control its behavior, but changes to a single CPU will sometimes
affect others (see `/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus`).

Some useful files:

```shell
/proc/cpuinfo
/sys/devices/system/cpu/possible
/sys/devices/system/cpu/present
/sys/devices/system/cpu/cpu0/online
/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies
/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq
/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus
/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
```

See the clockspeed of each CPU

```shell
$ for i in `cat /sys/devices/system/cpu/present | tr '-' ' ' | xargs seq`; do \
    paste \
      "/sys/devices/system/cpu/cpu${i?}/cpufreq/cpuinfo_cur_freq" \
      "/sys/devices/system/cpu/cpu${i?}/cpufreq/cpuinfo_min_freq" \
      "/sys/devices/system/cpu/cpu${i?}/cpufreq/cpuinfo_max_freq"; \
done
```

Before changing things, make sure to check the current scaling governor settings
first so you can put them back when you're done.

```shell
$ for i in `cat /sys/devices/system/cpu/present | tr '-' ' ' | xargs seq`; do \
    cat "/sys/devices/system/cpu/cpu${i?}/cpufreq/scaling_governor"; \
done
```

#### Single-Core Example

Here's an example to run IREE in a single-threaded context on CPU 7 at its
lowest clock speed.

First we'll take control of the clockspeed by setting the governor to
"userspace".

```shell
$ for i in `cat /sys/devices/system/cpu/present | tr '-' ' ' | xargs seq`; do \
  echo userspace > \
    "/sys/devices/system/cpu/cpu${i?}/cpufreq/scaling_governor"; \
done
```

We can now set individual clock speeds. We'll pin cpu7 to its minimum frequency.
We choose the minimum instead of the maximum here to mitigate thermal throttling
concerns

```shell
$ cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_min_freq > \
/sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed
```

We can confirm the frequencies of all the CPUs by running the same command
above. Now to run a command specifically on cpu7, use `taskset 80`
(hex for 10000000):

```shell
taskset 80 sleep 20 &
ps -o psr $!
```

Remember to cleanup when you're done! Here we'll set the scaling governor back
to schedutil because that's what they were before on the particular device this,
was tested on, but that may not exist on all devices.

```shell
$ for i in `cat /sys/devices/system/cpu/present | tr '-' ' ' | xargs seq`; do \
  echo schedutil > \
    "/sys/devices/system/cpu/cpu${i?}/cpufreq/scaling_governor"; \
done
```

#### Android Scripts

We provide a few scripts to set clockspeeds on Android (under
`build_tools/benchmarks`). These are somewhat device-specific:

* The `set_android_scaling_governor.sh` work on all CPUs, but the default
  governor name may be different across devices.
* The `set_*_gpu_scaling_policy.sh` script used should match the actual GPU on
  your device.

Sample configuration steps for Pixel 6:

1. Copy all scripts to the device:

   ```shell
   adb push build_tools/benchmarks/*.sh /data/local/tmp
   ```

1. Launch interactive adb shell as super user:

   ```shell
   adb shell
   oriole:/ # su
   oriole:/ # cd /data/local/tmp
   ```

1. Pin frequencies (high clockspeeds):

   ```shell
   oriole:/ # ./set_android_scaling_governor.sh
    CPU info (before changing governor):
    cpu     governor        cur     min     max
    ------------------------------------------------
    cpu0    sched_pixel     1098000 300000  1803000
    cpu1    sched_pixel     1598000 300000  1803000
    cpu2    sched_pixel     1598000 300000  1803000
    cpu3    sched_pixel     1098000 300000  1803000
    cpu4    sched_pixel     400000  400000  2253000
    cpu5    sched_pixel     400000  400000  2253000
    cpu6    sched_pixel     500000  500000  2802000
    cpu7    sched_pixel     500000  500000  2802000
    Setting CPU frequency governor to performance
    CPU info (after changing governor):
    cpu     governor        cur     min     max
    ------------------------------------------------
    cpu0    performance     1803000 300000  1803000
    cpu1    performance     1803000 300000  1803000
    cpu2    performance     1803000 300000  1803000
    cpu3    performance     1803000 300000  1803000
    cpu4    performance     2253000 400000  2253000
    cpu5    performance     2253000 400000  2253000
    cpu6    performance     2802000 500000  2802000
    cpu7    performance     2802000 500000  2802000
   oriole:/data/local/tmp # ./set_pixel6_gpu_scaling_policy.sh
    GPU info (before changing frequency scaling policy):
    policy                                  cur     min     max
    --------------------------------------------------------------
    coarse_demand [adaptive] always_on      251000  151000  848000
    Setting GPU frequency scaling policy to performance
    GPU info (after changing frequency scaling policy):
    policy                                  cur     min     max
    --------------------------------------------------------------
    coarse_demand adaptive [always_on]      848000  151000  848000
   ```

1. Restore default frequencies:

   ```shell
   oriole:/ # ./set_android_scaling_governor.sh sched_pixel
   ...
   oriole:/ # ./set_pixel6_gpu_scaling_policy.sh default
   ...
   ```

TODO(scotttodd): Windows instructions
