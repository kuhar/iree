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


### Replace the Dispatch

In this step, we will use a developer flag in iree-compile that allows for
specifying replacement binaried to be embeded in the final `.vmfb`, instead of
the compiled ones.

```shell
$ iree-compile <your compile args...> \
    --iree-hal-substitute-executable-object=main_0_dispatch_0=dump/module_main_0_dispatch_0_rocm_hsaco_fb.hsaco
```

