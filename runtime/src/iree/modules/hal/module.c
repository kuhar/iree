// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/hal/module.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

// Limit the number of bindings we pass down through the HAL. This can be tuned
// in the future but right now guards the stack from blowing up during calls.
#define IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT ((iree_host_size_t)32)

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

static iree_vm_ref_type_descriptor_t iree_hal_allocator_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_view_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_command_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_layout_descriptor =
    {0};
static iree_vm_ref_type_descriptor_t iree_hal_device_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_event_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_layout_descriptor = {
    0};
static iree_vm_ref_type_descriptor_t iree_hal_fence_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_semaphore_descriptor = {0};

#define IREE_VM_REGISTER_HAL_C_TYPE(type, name, destroy_fn, descriptor)   \
  descriptor.type_name = iree_make_cstring_view(name);                    \
  descriptor.offsetof_counter = offsetof(iree_hal_resource_t, ref_count); \
  descriptor.destroy = (iree_vm_ref_destroy_t)destroy_fn;                 \
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&descriptor));

IREE_API_EXPORT iree_status_t iree_hal_module_register_types(void) {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();

  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_allocator_t, "hal.allocator",
                              iree_hal_allocator_destroy,
                              iree_hal_allocator_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_t, "hal.buffer",
                              iree_hal_buffer_recycle,
                              iree_hal_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_view_t, "hal.buffer_view",
                              iree_hal_buffer_view_destroy,
                              iree_hal_buffer_view_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_command_buffer_t, "hal.command_buffer",
                              iree_hal_command_buffer_destroy,
                              iree_hal_command_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_descriptor_set_t, "hal.descriptor_set",
                              iree_hal_descriptor_set_destroy,
                              iree_hal_descriptor_set_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_descriptor_set_layout_t,
                              "hal.descriptor_set_layout",
                              iree_hal_descriptor_set_layout_destroy,
                              iree_hal_descriptor_set_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_device_t, "hal.device",
                              iree_hal_device_destroy,
                              iree_hal_device_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_event_t, "hal.event",
                              iree_hal_event_destroy,
                              iree_hal_event_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_executable_t, "hal.executable",
                              iree_hal_executable_destroy,
                              iree_hal_executable_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_executable_layout_t,
                              "hal.executable_layout",
                              iree_hal_executable_layout_destroy,
                              iree_hal_executable_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_fence_t, "hal.fence",
                              iree_hal_fence_destroy,
                              iree_hal_fence_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_semaphore_t, "hal.semaphore",
                              iree_hal_semaphore_destroy,
                              iree_hal_semaphore_descriptor);

  has_registered = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer_view, iree_hal_buffer_view_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_command_buffer,
                             iree_hal_command_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set,
                             iree_hal_descriptor_set_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set_layout,
                             iree_hal_descriptor_set_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_event, iree_hal_event_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable_layout,
                             iree_hal_executable_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_fence, iree_hal_fence_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_semaphore, iree_hal_semaphore_t);

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_hal_module_t {
  iree_allocator_t host_allocator;
  iree_hal_module_flags_t flags;
  iree_hal_device_t* shared_device;
  // TODO(benvanik): types.
} iree_hal_module_t;

#define IREE_HAL_MODULE_CAST(module) \
  (iree_hal_module_t*)((uint8_t*)(module) + iree_vm_native_module_size());

typedef struct iree_hal_module_state_t {
  iree_allocator_t host_allocator;

  // Flags controlling HAL module behavior passed in from the hosting
  // application. All instantiations of a module share the same flags.
  iree_hal_module_flags_t flags;

  // HACK: today we only support a single device per context - in the future
  // this should be a set of available devices that the module is able to pick
  // from - the module will then hang on to them and use them as native globals
  // instead of storing anything in module state here.
  iree_hal_device_t* shared_device;

  // TODO(benvanik): add iree_loop_t to module constructor.
  // Status of the nested loop we run for executable creation today. We should
  // instead be taking a loop upon creation and scheduling work against that.
  iree_status_t loop_status;

  // Shared executable cache for all executables created in the context.
  // We could have multiple to allow for modules to create distinct sets of
  // executables like ones for training vs inference in the same model, or just
  // always use this.
  iree_hal_executable_cache_t* executable_cache;

  // TODO(benvanik): remove with submit_and_wait.
  iree_hal_semaphore_t* submit_semaphore;
  uint64_t submit_value;
} iree_hal_module_state_t;

static void IREE_API_PTR iree_hal_module_destroy(void* base_module) {
  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(base_module);
  iree_hal_device_release(module->shared_device);
}

static iree_status_t IREE_API_PTR
iree_hal_module_alloc_state(void* self, iree_allocator_t host_allocator,
                            iree_vm_module_state_t** out_module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(self);
  iree_hal_module_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  state->flags = module->flags;
  state->shared_device = module->shared_device;
  iree_hal_device_retain(state->shared_device);

  state->loop_status = iree_ok_status();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_executable_cache_create(
              state->shared_device, iree_string_view_empty(),
              iree_loop_inline(&state->loop_status), &state->executable_cache));

  state->submit_value = 0ull;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_create(state->shared_device, state->submit_value,
                                    &state->submit_semaphore));

  *out_module_state = (iree_vm_module_state_t*)state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void IREE_API_PTR
iree_hal_module_free_state(void* self, iree_vm_module_state_t* module_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  iree_hal_semaphore_release(state->submit_semaphore);
  iree_hal_executable_cache_release(state->executable_cache);
  iree_status_ignore(state->loop_status);
  iree_hal_device_release(state->shared_device);
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t IREE_API_PTR iree_hal_module_notify(
    void* self, iree_vm_module_state_t* module_state, iree_vm_signal_t signal) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  switch (signal) {
    case IREE_VM_SIGNAL_SUSPEND:
    case IREE_VM_SIGNAL_LOW_MEMORY:
      return iree_hal_device_trim(state->shared_device);
    default:
      return iree_ok_status();
  }
}

//===----------------------------------------------------------------------===//
// Experimental APIs
//===----------------------------------------------------------------------===//
// NOTE: Ex* APIs are experimental and likely to be removed soon. Modules
// using these APIs are not forward compatible.

IREE_VM_ABI_EXPORT(iree_hal_module_ex_shared_device,  //
                   iree_hal_module_state_t,           //
                   v, r) {
  rets->r0 = iree_hal_device_retain_ref(state->shared_device);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_ex_submit_and_wait,  //
                   iree_hal_module_state_t,             //
                   rr, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r1, &command_buffer));

  // Batch with our single command buffer.
  iree_hal_submission_batch_t batch;
  memset(&batch, 0, sizeof(batch));

  iree_hal_command_buffer_t* command_buffer_ptrs[] = {command_buffer};
  batch.command_buffer_count = IREE_ARRAYSIZE(command_buffer_ptrs);
  batch.command_buffers = command_buffer_ptrs;

  uint64_t next_semaphore_value = ++state->submit_value;
  iree_hal_semaphore_t* signal_semaphore_ptrs[] = {state->submit_semaphore};
  uint64_t signal_semaphore_values[] = {next_semaphore_value};
  batch.signal_semaphores.count = IREE_ARRAYSIZE(signal_semaphore_ptrs);
  batch.signal_semaphores.semaphores = signal_semaphore_ptrs;
  batch.signal_semaphores.payload_values = signal_semaphore_values;

  iree_status_t status = iree_hal_device_queue_submit(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, 0, 1, &batch);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        state->submit_semaphore, next_semaphore_value, iree_infinite_timeout());
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Casts a VM value to a C host size.
static iree_host_size_t iree_hal_cast_host_size(int64_t value) {
  // TODO(benvanik): make this return status and check for overflow if host
  // size is 32-bits.
  return (iree_host_size_t)value;
}

// Casts a VM value to a HAL device size.
static iree_device_size_t iree_hal_cast_device_size(int64_t value) {
  // TODO(benvanik): make this return status and check for overflow if device
  // size is 32-bits.
  return (iree_device_size_t)value;
}

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_allocator_allocate,  //
                   iree_hal_module_state_t,             //
                   riiI, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i1;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i2;
  iree_device_size_t allocation_size = iree_hal_cast_device_size(args->i3);

  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, iree_const_byte_span_empty(),
      &buffer));
  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_allocator_allocate_initialized,  //
                   iree_hal_module_state_t,                         //
                   riirII, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i1;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i2;
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r3, &source));
  iree_device_size_t offset = iree_hal_cast_device_size(args->i4);
  iree_device_size_t length = iree_hal_cast_device_size(args->i5);

  iree_host_size_t buffer_length = source->data.data_length;
  if (length == -1) {
    length = buffer_length;
  }
  if (length < 0 || offset < 0 || offset > buffer_length ||
      offset + length > buffer_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "byte range out of bounds (requested %" PRIdsz
                            "-%" PRIdsz " of available %" PRIhsz ")",
                            offset, (offset + length - 1), buffer_length);
  }

  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(
          allocator, params, length,
          iree_make_const_byte_span(source->data.data + offset, length),
          &buffer),
      "failed to allocate buffer of length %" PRIdsz, length);

  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

static void iree_hal_module_mapped_buffer_release(void* user_data,
                                                  iree_hal_buffer_t* buffer) {
  iree_vm_buffer_t* backing_buffer = (iree_vm_buffer_t*)user_data;
  iree_vm_buffer_release(backing_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_allocator_map_byte_buffer,  //
                   iree_hal_module_state_t,                    //
                   riiirII, r) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r0, &allocator));
  bool is_try = args->i1 != 0;
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i2;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i3;
  iree_vm_buffer_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r4, &source));
  iree_device_size_t offset = iree_hal_cast_device_size(args->i5);
  iree_device_size_t length = iree_hal_cast_device_size(args->i6);

  iree_host_size_t buffer_length = source->data.data_length;
  if (length == -1) {
    length = buffer_length;
  }
  if (length < 0 || offset < 0 || offset > buffer_length ||
      offset + length > buffer_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "byte range out of bounds (requested %" PRIdsz
                            "-%" PRIdsz " of available %" PRIhsz ")",
                            offset, (offset + length - 1), buffer_length);
  }

  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_READ;
  if (!iree_all_bits_set(source->access, IREE_VM_BUFFER_ACCESS_MUTABLE)) {
    // Source buffer is read-only; require that the access request matches.
    if (!iree_all_bits_set(buffer_usage,
                           IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE)) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "source buffer is immutable and can only be "
                              "mapped for constant usage");
    }

    // NOTE: if we wanted to lock things down for when there's no MMU to ensure
    // that the loaded program doesn't touch the memory then we could just fail
    // the request - the program will then perform an alloc+copy and can do
    // whatever it wants with the memory.
  } else {
    // Source buffer is mutable; allow in-place writes.
    if (!iree_all_bits_set(buffer_usage,
                           IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE)) {
      allowed_access |= IREE_HAL_MEMORY_ACCESS_WRITE;
    }
  }

  // Try mapping - note that this may fail if the target device cannot map the
  // memory into the given type (for example, mapping a host buffer into
  // device-local memory is only going to work on unified memory systems).
  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
      .access = allowed_access,
  };
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = length,
      .handle.host_allocation.ptr = source->data.data + offset,
  };
  iree_hal_buffer_release_callback_t release_callback = {
      .fn = iree_hal_module_mapped_buffer_release,
      .user_data = source,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_import_buffer(
      allocator, params, &external_buffer, release_callback, &buffer);
  if (iree_status_is_ok(status)) {
    // Mapping succeeded - retain the source buffer that'll be released by
    // iree_hal_module_map_data_ctl when the mapping is no longer used.
    iree_vm_buffer_retain(source);
    rets->r0 = iree_hal_buffer_move_ref(buffer);
    return iree_ok_status();
  }

  // Failed to map - if this was a try then don't fail and just rely on the
  // result being nullptr to indicate to the caller that things failed.
  memset(&rets->r0, 0, sizeof(rets->r0));
  if (is_try) {
    iree_status_ignore(status);
    return iree_ok_status();
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_assert,  //
                   iree_hal_module_state_t,        //
                   rrrIii, v) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  iree_vm_buffer_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &message));
  iree_string_view_t message_str IREE_ATTRIBUTE_UNUSED =
      iree_vm_buffer_as_string(message);
  iree_hal_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_check_deref(args->r2, &allocator));
  iree_device_size_t minimum_length = iree_hal_cast_device_size(args->i3);
  iree_hal_memory_type_t required_memory_types =
      (iree_hal_memory_type_t)args->i4;
  iree_hal_buffer_usage_t required_buffer_usage =
      (iree_hal_buffer_usage_t)args->i5;

  // Ensure we have enough bytes in the buffer for the encoding we have.
  // Note that having more bytes is fine:
  //   assert(expected_length <= actual_length);
  iree_device_size_t actual_length = iree_hal_buffer_byte_length(buffer);
  if (actual_length < minimum_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s buffer byte length %" PRIdsz
                            " less than expected minimum %" PRIdsz,
                            (int)message_str.size, message_str.data,
                            actual_length, minimum_length);
  }

  // TODO(benvanik): assert that the buffer view is accessible from the
  // target device. This needs some iree_hal_allocator_* methods for checking
  // whether the external buffer can be used. To start we just compare if the
  // allocators are identical.

  // All memory type bits expected (indicating where the program intends to use
  // the buffer data) must be set in the buffer while the buffer is allowed to
  // have more bits.
  iree_hal_memory_type_t actual_memory_type =
      iree_hal_buffer_memory_type(buffer);
  if (!iree_all_bits_set(actual_memory_type, required_memory_types)) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t actual_memory_type_str =
        iree_hal_memory_type_format(actual_memory_type, &temp0);
    iree_string_view_t expected_memory_type_str =
        iree_hal_memory_type_format(required_memory_types, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s buffer memory type is not compatible; buffer has %.*s, operation "
        "requires %.*s",
        (int)message_str.size, message_str.data,
        (int)actual_memory_type_str.size, actual_memory_type_str.data,
        (int)expected_memory_type_str.size, expected_memory_type_str.data);
#else
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s buffer memory type is not compatible; buffer has %08X, operation "
        "requires %08X",
        (int)message_str.size, message_str.data, actual_memory_type,
        expected_memory_type);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  // All usage bits expected (indicating what the program intends to use the
  // buffer for) must be set in the buffer while the buffer is allowed to have
  // more bits.
  iree_hal_buffer_usage_t actual_buffer_usage =
      iree_hal_buffer_allowed_usage(buffer);
  if (!iree_all_bits_set(actual_buffer_usage, required_buffer_usage)) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_usage_str =
        iree_hal_buffer_usage_format(actual_buffer_usage, &temp0);
    iree_string_view_t required_usage_str =
        iree_hal_buffer_usage_format(required_buffer_usage, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s requested usage was not specified when the buffer was allocated; "
        "buffer allows %.*s, operation requires %.*s",
        (int)message_str.size, message_str.data, (int)allowed_usage_str.size,
        allowed_usage_str.data, (int)required_usage_str.size,
        required_usage_str.data);
#else
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "%.*s requested usage was not specified when the buffer was allocated; "
        "buffer allows %08X, operation requires %08X",
        (int)message_str.size, message_str.data, allowed_buffer_usage,
        required_buffer_usage);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_subspan,  //
                   iree_hal_module_state_t,         //
                   rII, r) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i1);
  iree_device_size_t length = iree_hal_cast_device_size(args->i2);

  iree_hal_buffer_t* subspan_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_subspan(source_buffer, source_offset, length,
                              &subspan_buffer),
      "invalid subspan of an existing buffer (source_offset=%" PRIdsz
      ", length=%" PRIdsz ")",
      source_offset, length);
  rets->r0 = iree_hal_buffer_move_ref(subspan_buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_length,  //
                   iree_hal_module_state_t,        //
                   r, I) {
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &buffer));
  rets->i0 = (int64_t)iree_hal_buffer_byte_length(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_load,  //
                   iree_hal_module_state_t,      //
                   rIi, i) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i1);
  iree_vm_size_t length = (iree_vm_size_t)args->i2;

  uint32_t target_buffer = 0;
  if (length > sizeof(target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "load length byte count %d exceeds max", length);
  }

  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      state->shared_device, source_buffer, source_offset, &target_buffer,
      length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  rets->i0 = target_buffer;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_store,  //
                   iree_hal_module_state_t,       //
                   irIi, v) {
  int32_t value = args->i0;
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i2);
  iree_vm_size_t length = (iree_vm_size_t)args->i3;

  if (length > sizeof(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "store length byte count %d exceeds max", length);
  } else if (target_offset + length >
             iree_hal_buffer_byte_length(target_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "store out of bounds (target_offset=%" PRIdsz
                            ", length=%d into max %" PRIdsz ")",
                            target_offset, length,
                            iree_hal_buffer_byte_length(target_buffer));
  }

  return iree_hal_device_transfer_h2d(
      state->shared_device, &value, target_buffer, target_offset, length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_create,  //
                   iree_hal_module_state_t,             //
                   riiCID, r) {
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r0, &source_buffer));
  iree_hal_element_type_t element_type = (iree_hal_element_type_t)args->i1;
  iree_hal_encoding_type_t encoding_type = (iree_hal_encoding_type_t)args->i2;
  iree_host_size_t shape_rank = 0;
  iree_hal_dim_t* shape_dims = NULL;
  // TODO(benvanik): avoid the cast/alloca if not required.
  IREE_VM_ABI_VLA_STACK_CAST(args, a3_count, a3, iree_hal_dim_t, 128,
                             &shape_rank, &shape_dims);

  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      source_buffer, shape_rank, shape_dims, element_type, encoding_type,
      state->host_allocator, &buffer_view));
  rets->r0 = iree_hal_buffer_view_move_ref(buffer_view);
  return iree_ok_status();
}

// Returns true if the |expected_type| can be satisfied with |actual_type|.
// This allows for basic type widening and bypassing instead of requiring an
// exact match in all cases.
static bool iree_hal_element_types_are_compatible(
    iree_hal_element_type_t actual_type,
    iree_hal_element_type_t expected_type) {
  if (iree_hal_element_numerical_type_is_opaque(actual_type)) {
    // If the provided type is opaque it can map to anything. This allows
    // applications to bypass the checks when they are treating all the data as
    // opaque, such as when carrying around buffer data in binary blobs.
    return true;
  }

  if (iree_hal_element_numerical_type_is_integer(actual_type) &&
      iree_hal_element_numerical_type_is_integer(expected_type) &&
      iree_hal_element_bit_count(actual_type) ==
          iree_hal_element_bit_count(expected_type)) {
    // Integer types of the same bit width are allowed to be cast.
    // This allows users or the compiler to treat data as signless while still
    // allowing signedness. For example, tensor<1xi32> can successfully match
    // a tensor<1xui32> expectation.
    return true;
  }

  // Otherwise we require an exact match. This may be overly conservative but
  // in most cases is a useful error message. Users can pass in OPAQUE types if
  // hitting this to bypass.
  return actual_type == expected_type;
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_assert,  //
                   iree_hal_module_state_t,             //
                   rriiCID, v) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  iree_vm_buffer_t* message = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &message));
  iree_string_view_t message_str IREE_ATTRIBUTE_UNUSED =
      iree_vm_buffer_as_string(message);
  iree_hal_element_type_t expected_element_type =
      (iree_hal_element_type_t)args->i2;
  iree_hal_encoding_type_t expected_encoding_type =
      (iree_hal_encoding_type_t)args->i3;
  iree_host_size_t expected_shape_rank = 0;
  iree_hal_dim_t* expected_shape_dims = NULL;
  // TODO(benvanik): avoid the cast/alloca if not required.
  IREE_VM_ABI_VLA_STACK_CAST(args, a4_count, a4, iree_hal_dim_t, 128,
                             &expected_shape_rank, &expected_shape_dims);

  // Check encoding first; getting the encoding wrong is worse than the shape.
  // If the actual encoding is opaque we allow it to pass through - this lets
  // users override the assertion in the case where they are just passing data
  // around and don't care about the contents.
  iree_hal_encoding_type_t actual_encoding_type =
      iree_hal_buffer_view_encoding_type(buffer_view);
  if (actual_encoding_type != IREE_HAL_ENCODING_TYPE_OPAQUE &&
      actual_encoding_type != expected_encoding_type) {
    // TODO(benvanik): string formatting of encodings.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s encoding mismatch; expected %08X but have %08X",
        (int)message_str.size, message_str.data, expected_encoding_type,
        actual_encoding_type);
  }

  // Element types determine the storage requirements.
  // If the actual element type is opaque we allow it to pass through.
  iree_hal_element_type_t actual_element_type =
      iree_hal_buffer_view_element_type(buffer_view);
  if (!iree_hal_element_types_are_compatible(actual_element_type,
                                             expected_element_type)) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
    char actual_element_type_str[32];
    iree_host_size_t actual_element_type_str_length = 0;
    char expected_element_type_str[32];
    iree_host_size_t expected_element_type_str_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
        actual_element_type, sizeof(actual_element_type_str),
        actual_element_type_str, &actual_element_type_str_length));
    IREE_RETURN_IF_ERROR(iree_hal_format_element_type(
        expected_element_type, sizeof(expected_element_type_str),
        expected_element_type_str, &expected_element_type_str_length));
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s element type mismatch; expected %.*s (%08X) but have %.*s (%08X)",
        (int)message_str.size, message_str.data,
        (int)expected_element_type_str_length, expected_element_type_str,
        expected_element_type, (int)actual_element_type_str_length,
        actual_element_type_str, actual_element_type);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s element type mismatch; expected %08X but have %08X",
        (int)message_str.size, message_str.data, expected_element_type,
        actual_element_type);
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  }

  // Rank check before the individual shape dimensions.
  iree_host_size_t actual_shape_rank =
      iree_hal_buffer_view_shape_rank(buffer_view);
  const iree_hal_dim_t* actual_shape_dims =
      iree_hal_buffer_view_shape_dims(buffer_view);
  iree_status_t shape_status = iree_ok_status();
  if (actual_shape_rank != expected_shape_rank) {
    shape_status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s shape rank mismatch; expected %" PRIhsz " but have %" PRIhsz,
        (int)message_str.size, message_str.data, expected_shape_rank,
        actual_shape_rank);
  }
  if (iree_status_is_ok(shape_status)) {
    for (iree_host_size_t i = 0; i < actual_shape_rank; ++i) {
      if (actual_shape_dims[i] == expected_shape_dims[i]) continue;
      // Dimension mismatch.
      shape_status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "%.*s shape dimension %" PRIhsz
                           " mismatch; expected %" PRIdim " but have %" PRIdim,
                           (int)message_str.size, message_str.data, i,
                           expected_shape_dims[i], actual_shape_dims[i]);
      break;
    }
  }

#if IREE_HAL_MODULE_STRING_UTIL_ENABLE
  if (!iree_status_is_ok(shape_status)) {
    char actual_shape_str[32];
    iree_host_size_t actual_shape_str_length = 0;
    char expected_shape_str[32];
    iree_host_size_t expected_shape_str_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_format_shape(
        actual_shape_rank, actual_shape_dims, sizeof(actual_shape_str),
        actual_shape_str, &actual_shape_str_length));
    IREE_RETURN_IF_ERROR(iree_hal_format_shape(
        expected_shape_rank, expected_shape_dims, sizeof(expected_shape_str),
        expected_shape_str, &expected_shape_str_length));
    shape_status = iree_status_annotate_f(
        shape_status, "expected shape %.*s, actual shape %.*s",
        (int)expected_shape_str_length, expected_shape_str,
        (int)actual_shape_str_length, actual_shape_str);
  }
#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE

  return shape_status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_buffer,  //
                   iree_hal_module_state_t,             //
                   r, r) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->r0 =
      iree_hal_buffer_retain_ref(iree_hal_buffer_view_buffer(buffer_view));
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_element_type,  //
                   iree_hal_module_state_t,                   //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (uint32_t)iree_hal_buffer_view_element_type(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_encoding_type,  //
                   iree_hal_module_state_t,                    //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (uint32_t)iree_hal_buffer_view_encoding_type(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_rank,  //
                   iree_hal_module_state_t,           //
                   r, i) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  rets->i0 = (iree_vm_size_t)iree_hal_buffer_view_shape_rank(buffer_view);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_dim,  //
                   iree_hal_module_state_t,          //
                   ri, I) {
  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_view_check_deref(args->r0, &buffer_view));
  iree_vm_size_t index = (iree_vm_size_t)args->i1;
  rets->i0 = (int64_t)iree_hal_buffer_view_shape_dim(buffer_view, index);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_buffer_view_trace,  //
                   iree_hal_module_state_t,            //
                   rCrD, v) {
#if IREE_HAL_MODULE_STRING_UTIL_ENABLE

  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r0, &key));
  iree_string_view_t key_str = iree_vm_buffer_as_string(key);

  fprintf(stderr, "=== %.*s ===\n", (int)key_str.size, key_str.data);
  for (iree_host_size_t i = 0; i < args->a1_count; ++i) {
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_check_deref(args->a1[i].r0, &buffer_view));

    // NOTE: this export is for debugging only and a no-op in min-size builds.
    // We heap-alloc here because at the point this export is used performance
    // is not a concern.

    // Query total length (excluding NUL terminator).
    iree_host_size_t result_length = 0;
    iree_status_t status = iree_hal_buffer_view_format(buffer_view, SIZE_MAX, 0,
                                                       NULL, &result_length);
    if (!iree_status_is_out_of_range(status)) {
      return status;
    }
    ++result_length;  // include NUL

    // Allocate scratch heap memory to contain the result and format into it.
    char* result_str = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        state->host_allocator, result_length, (void**)&result_str));
    status = iree_hal_buffer_view_format(buffer_view, SIZE_MAX, result_length,
                                         result_str, &result_length);
    if (iree_status_is_ok(status)) {
      fprintf(stderr, "%.*s\n", (int)result_length, result_str);
    }
    iree_allocator_free(state->host_allocator, result_str);
    IREE_RETURN_IF_ERROR(status);
  }
  fprintf(stderr, "\n");

#endif  // IREE_HAL_MODULE_STRING_UTIL_ENABLE
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_create,  //
                   iree_hal_module_state_t,                //
                   rii, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_command_buffer_mode_t modes =
      (iree_hal_command_buffer_mode_t)args->i1;
  iree_hal_command_category_t command_categories =
      (iree_hal_command_category_t)args->i2;

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device, modes, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  iree_status_t status = iree_hal_command_buffer_begin(command_buffer);
  if (iree_status_is_ok(status)) {
    rets->r0 = iree_hal_command_buffer_move_ref(command_buffer);
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_finalize,  //
                   iree_hal_module_state_t,                  //
                   r, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));

  return iree_hal_command_buffer_end(command_buffer);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_begin_debug_group,  //
                   iree_hal_module_state_t,                           //
                   rr, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_vm_buffer_t* label = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &label));
  iree_string_view_t label_str = iree_vm_buffer_as_string(label);
  // TODO(benvanik): query from VM.
  iree_hal_label_location_t location = {
      .file = iree_string_view_empty(),
      .line = 0,
  };
  iree_hal_command_buffer_begin_debug_group(
      command_buffer, label_str, iree_hal_label_color_unspecified(), &location);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_end_debug_group,  //
                   iree_hal_module_state_t,                         //
                   r, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_command_buffer_end_debug_group(command_buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_execution_barrier,  //
                   iree_hal_module_state_t,                           //
                   riii, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_execution_stage_t source_stage_mask =
      (iree_hal_execution_stage_t)args->i1;
  iree_hal_execution_stage_t target_stage_mask =
      (iree_hal_execution_stage_t)args->i2;
  iree_hal_execution_barrier_flags_t flags =
      (iree_hal_execution_barrier_flags_t)args->i3;

  // TODO(benvanik): decode barriers.
  iree_hal_memory_barrier_t global_barrier;
  global_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  global_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;

  return iree_hal_command_buffer_execution_barrier(
      command_buffer, source_stage_mask, target_stage_mask, flags, 1,
      &global_barrier, 0, NULL);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_fill_buffer,  //
                   iree_hal_module_state_t,                     //
                   rrIIii, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i2);
  iree_device_size_t length = iree_hal_cast_device_size(args->i3);
  uint32_t pattern = (uint32_t)args->i4;
  uint32_t pattern_length = (uint32_t)args->i5;

  return iree_hal_command_buffer_fill_buffer(command_buffer, target_buffer,
                                             target_offset, length, &pattern,
                                             pattern_length);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_copy_buffer,  //
                   iree_hal_module_state_t,                     //
                   rrIrII, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_buffer_t* source_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r1, &source_buffer));
  iree_device_size_t source_offset = iree_hal_cast_device_size(args->i2);
  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r3, &target_buffer));
  iree_device_size_t target_offset = iree_hal_cast_device_size(args->i4);
  iree_device_size_t length = iree_hal_cast_device_size(args->i5);

  return iree_hal_command_buffer_copy_buffer(command_buffer, source_buffer,
                                             source_offset, target_buffer,
                                             target_offset, length);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_push_constants,  //
                   iree_hal_module_state_t,                        //
                   rriCiD, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_layout_check_deref(args->r1, &executable_layout));
  iree_vm_size_t offset = (iree_vm_size_t)args->i2;
  iree_host_size_t value_count = args->a3_count;
  const uint32_t* values = (const uint32_t*)&args->a3[0].i0;

  return iree_hal_command_buffer_push_constants(
      command_buffer, executable_layout, offset * sizeof(uint32_t), values,
      value_count * sizeof(uint32_t));
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_push_descriptor_set,  //
                   iree_hal_module_state_t,                             //
                   rriCirIID, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_layout_check_deref(args->r1, &executable_layout));
  iree_vm_size_t set = args->i2;

  iree_host_size_t binding_count = args->a3_count;
  if (IREE_UNLIKELY(binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE, "binding count %" PRIhsz " > %" PRIhsz,
        binding_count, IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_descriptor_set_binding_t* bindings =
      (iree_hal_descriptor_set_binding_t*)iree_alloca(
          binding_count * sizeof(iree_hal_descriptor_set_binding_t));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_check_deref(args->a3[i].r1, &bindings[i].buffer));
    bindings[i].binding = (uint32_t)args->a3[i].i0;
    bindings[i].offset = iree_hal_cast_device_size(args->a3[i].i2);
    bindings[i].length = iree_hal_cast_device_size(args->a3[i].i3);
  }

  return iree_hal_command_buffer_push_descriptor_set(
      command_buffer, executable_layout, set, binding_count, bindings);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_bind_descriptor_set,  //
                   iree_hal_module_state_t,                             //
                   rrirCID, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_layout_check_deref(args->r1, &executable_layout));
  int32_t set = args->i2;
  iree_hal_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_descriptor_set_check_deref(args->r3, &descriptor_set));
  iree_host_size_t dynamic_offset_count = 0;
  iree_device_size_t* dynamic_offsets = NULL;
  // TODO(benvanik): avoid the cast/alloca if not required.
  IREE_VM_ABI_VLA_STACK_CAST(args, a4_count, a4, iree_device_size_t, 64,
                             &dynamic_offset_count, &dynamic_offsets);
  return iree_hal_command_buffer_bind_descriptor_set(
      command_buffer, executable_layout, set, descriptor_set,
      dynamic_offset_count, dynamic_offsets);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_dispatch,  //
                   iree_hal_module_state_t,                  //
                   rriiii, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_check_deref(args->r1, &executable));
  uint32_t entry_point = (uint32_t)args->i2;
  uint32_t workgroup_x = (uint32_t)args->i3;
  uint32_t workgroup_y = (uint32_t)args->i4;
  uint32_t workgroup_z = (uint32_t)args->i5;
  return iree_hal_command_buffer_dispatch(command_buffer, executable,
                                          entry_point, workgroup_x, workgroup_y,
                                          workgroup_z);
}

IREE_VM_ABI_EXPORT(iree_hal_module_command_buffer_dispatch_indirect,  //
                   iree_hal_module_state_t,                           //
                   rrirI, v) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_check_deref(args->r0, &command_buffer));
  iree_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_check_deref(args->r1, &executable));
  uint32_t entry_point = (uint32_t)args->i2;
  iree_hal_buffer_t* workgroups_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_check_deref(args->r3, &workgroups_buffer));
  iree_device_size_t workgroups_offset = (iree_vm_size_t)args->i4;
  return iree_hal_command_buffer_dispatch_indirect(
      command_buffer, executable, entry_point, workgroups_buffer,
      workgroups_offset);
}

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_descriptor_set_create,  //
                   iree_hal_module_state_t,                //
                   rrCirIID, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_descriptor_set_layout_t* set_layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_descriptor_set_layout_check_deref(args->r1, &set_layout));

  iree_host_size_t binding_count = args->a2_count;
  if (IREE_UNLIKELY(binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE, "binding count %" PRIhsz " > %" PRIhsz,
        binding_count, IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_descriptor_set_binding_t* bindings =
      (iree_hal_descriptor_set_binding_t*)iree_alloca(
          binding_count * sizeof(iree_hal_descriptor_set_binding_t));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_check_deref(args->a2[i].r1, &bindings[i].buffer));
    bindings[i].binding = (uint32_t)args->a2[i].i0;
    bindings[i].offset = iree_hal_cast_device_size(args->a2[i].i2);
    bindings[i].length = iree_hal_cast_device_size(args->a2[i].i3);
  }

  iree_hal_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_create(
      device, set_layout, binding_count, bindings, &descriptor_set));
  rets->r0 = iree_hal_descriptor_set_move_ref(descriptor_set);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_layout
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_descriptor_set_layout_create,  //
                   iree_hal_module_state_t,                       //
                   riCiiD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_descriptor_set_layout_usage_type_t usage_type =
      (iree_hal_descriptor_set_layout_usage_type_t)args->i1;

  iree_host_size_t binding_count = args->a2_count;
  if (IREE_UNLIKELY(binding_count >
                    IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE, "binding count %" PRIhsz " > %" PRIhsz,
        binding_count, IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT);
  }
  iree_hal_descriptor_set_layout_binding_t* bindings =
      (iree_hal_descriptor_set_layout_binding_t*)iree_alloca(
          binding_count * sizeof(iree_hal_descriptor_set_layout_binding_t));
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    bindings[i].binding = (uint32_t)args->a2[i].i0;
    bindings[i].type = (iree_hal_descriptor_type_t)args->a2[i].i1;
  }

  iree_hal_descriptor_set_layout_t* descriptor_set_layout = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_layout_create(
      device, usage_type, binding_count, bindings, &descriptor_set_layout));
  rets->r0 = iree_hal_descriptor_set_layout_move_ref(descriptor_set_layout);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_device_allocator,  //
                   iree_hal_module_state_t,           //
                   r, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  rets->r0 = iree_hal_allocator_retain_ref(iree_hal_device_allocator(device));
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_query_i64,  //
                   iree_hal_module_state_t,           //
                   rrr, iI) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_vm_buffer_t* category = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r1, &category));
  iree_string_view_t category_str = iree_vm_buffer_as_string(category);
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r2, &key));
  iree_string_view_t key_str = iree_vm_buffer_as_string(key);

  int64_t value = 0;
  iree_status_t query_status =
      iree_hal_device_query_i64(device, category_str, key_str, &value);
  rets->i0 = iree_status_consume_code(query_status) == IREE_STATUS_OK ? 1 : 0;
  rets->i1 = value;
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_alloca,  //
                   iree_hal_module_state_t,              //
                   rIrriiiI, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  uint32_t pool = args->i4;
  iree_hal_memory_type_t memory_types = (iree_hal_memory_type_t)args->i5;
  iree_hal_buffer_usage_t buffer_usage = (iree_hal_buffer_usage_t)args->i6;
  iree_device_size_t allocation_size = iree_hal_cast_device_size(args->i7);

  // TODO(benvanik): HAL APIs for queue-ordered allocations.
  // For now we just perform a blocking wait to synchronize with the queue,
  // allocate the buffer as normal, and then pass it back committed.
  (void)queue_affinity;
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_wait(wait_fence, iree_infinite_timeout()));

  // TODO(benvanik): enforce queue-ordered allocation restrictions on memory
  // type and usage.
  (void)pool;

  const iree_hal_buffer_params_t params = {
      .type = memory_types,
      .usage = buffer_usage,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, allocation_size,
      iree_const_byte_span_empty(), &buffer));

  IREE_RETURN_IF_ERROR(iree_hal_fence_signal(signal_fence));

  rets->r0 = iree_hal_buffer_move_ref(buffer);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_dealloca,  //
                   iree_hal_module_state_t,                //
                   rIrrr, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_check_deref(args->r4, &buffer));

  // TODO(benvanik): HAL APIs for queue-ordered allocations.
  // For now we just perform a blocking wait to synchronize with the queue and
  // then ignore the buffer for GC to cleanup.
  (void)queue_affinity;
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_wait(wait_fence, iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(iree_hal_fence_signal(signal_fence));

  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_execute,  //
                   iree_hal_module_state_t,               //
                   rIrrCrD, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;
  iree_hal_fence_t* wait_fence = iree_hal_fence_deref(args->r2);
  iree_hal_fence_t* signal_fence = iree_hal_fence_deref(args->r3);
  iree_host_size_t command_buffer_count = 0;
  iree_hal_command_buffer_t** command_buffers = NULL;
  IREE_VM_ABI_VLA_STACK_DEREF(args, a4_count, a4, iree_hal_command_buffer, 32,
                              &command_buffer_count, &command_buffers);

  iree_hal_submission_batch_t batch = {
      .wait_semaphores = iree_hal_fence_semaphore_list(wait_fence),
      .signal_semaphores = iree_hal_fence_semaphore_list(signal_fence),
      .command_buffer_count = command_buffer_count,
      .command_buffers = command_buffers,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_submit(
      device, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity, 1, &batch));

  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_device_queue_flush,  //
                   iree_hal_module_state_t,             //
                   rI, v) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_hal_queue_affinity_t queue_affinity =
      (iree_hal_queue_affinity_t)args->i1;

  // TODO(benvanik): queue flush API.
  // This will be most useful for backends that perform internal batching and
  // require the explicit flush. For now we don't have this exposed.
  (void)device;
  (void)queue_affinity;

  return iree_ok_status();
}

//===--------------------------------------------------------------------===//
// iree_hal_executable_t
//===--------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_executable_create,  //
                   iree_hal_module_state_t,            //
                   rrrrCrD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  iree_vm_buffer_t* executable_format = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_check_deref(args->r1, &executable_format));
  iree_string_view_t executable_format_str =
      iree_vm_buffer_as_string(executable_format);
  iree_vm_buffer_t* executable_data = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_check_deref(args->r2, &executable_data));
  iree_host_size_t constant_count = 0;
  const uint32_t* constants = NULL;
  if (iree_vm_buffer_isa(args->r3)) {
    iree_vm_buffer_t* constant_buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_check_deref(args->r3, &constant_buffer));
    if (constant_buffer->data.data_length % 4 != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "constant buffer data must contain 4-byte "
                              "elements but data length is %" PRIhsz,
                              constant_buffer->data.data_length);
    }
    constant_count = constant_buffer->data.data_length / sizeof(uint32_t);
    constants = (const uint32_t*)constant_buffer->data.data;
  }
  iree_host_size_t executable_layout_count = args->a4_count;
  iree_hal_executable_layout_t** executable_layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator,
      executable_layout_count * sizeof(executable_layouts[0]),
      (void**)&executable_layouts));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < executable_layout_count; ++i) {
    status = iree_hal_executable_layout_check_deref(args->a4[i].r0,
                                                    &executable_layouts[i]);
    if (!iree_status_is_ok(status)) break;
  }

  iree_hal_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode |=
        executable_data->access == IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE
            ? IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA
            : 0;
    executable_params.executable_format = executable_format_str;
    executable_params.executable_data = iree_make_const_byte_span(
        executable_data->data.data, executable_data->data.data_length);
    executable_params.executable_layout_count = executable_layout_count;
    executable_params.executable_layouts = executable_layouts;
    executable_params.constant_count = constant_count;
    executable_params.constants = constants;
    status = iree_hal_executable_cache_prepare_executable(
        state->executable_cache, &executable_params, &executable);
  }

  iree_allocator_free(state->host_allocator, executable_layouts);
  rets->r0 = iree_hal_executable_move_ref(executable);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_executable_layout_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_executable_layout_create,  //
                   iree_hal_module_state_t,                   //
                   riCrD, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  int32_t push_constants = (int32_t)args->i1;
  iree_host_size_t set_layout_count = 0;
  iree_hal_descriptor_set_layout_t** set_layouts = NULL;
  IREE_VM_ABI_VLA_STACK_DEREF(args, a2_count, a2,
                              iree_hal_descriptor_set_layout, 32,
                              &set_layout_count, &set_layouts);

  iree_hal_executable_layout_t* executable_layout = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_layout_create(
      device, push_constants, set_layout_count, set_layouts,
      &executable_layout));
  rets->r0 = iree_hal_executable_layout_move_ref(executable_layout);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_fence_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_fence_create,  //
                   iree_hal_module_state_t,       //
                   CrID, r) {
  // Create fence with enough capacity to store all the timepoints.
  // The count may end up lower if some are deduplicated.
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_create(args->a0_count, state->host_allocator, &fence));

  // Insert each timepoint into the fence.
  // This will deduplicate the semaphores and max their values.
  // The compiler already does this but we want to ensure that invariant is met
  // and don't trust the user code - at the point we'd have to verify
  // correctness it's easier just to use the same code path as insertion.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < args->a0_count; ++i) {
    iree_hal_semaphore_t* semaphore = NULL;
    status = iree_hal_semaphore_check_deref(args->a0[i].r0, &semaphore);
    if (!iree_status_is_ok(status)) break;
    uint64_t min_value = args->a0[i].i1;
    status = iree_hal_fence_insert(fence, semaphore, min_value);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    rets->r0 = iree_hal_fence_move_ref(fence);
  } else {
    iree_hal_fence_release(fence);
  }
  return status;
}

IREE_VM_ABI_EXPORT(iree_hal_module_fence_join,  //
                   iree_hal_module_state_t,     //
                   CrD, r) {
  iree_host_size_t fence_count = 0;
  iree_hal_fence_t** fences = NULL;
  IREE_VM_ABI_VLA_STACK_DEREF(args, a0_count, a0, iree_hal_fence, 32,
                              &fence_count, &fences);

  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_join(fence_count, fences, state->host_allocator, &fence));

  rets->r0 = iree_hal_fence_move_ref(fence);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_fence_signal,  //
                   iree_hal_module_state_t,       //
                   r, v) {
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_fence_check_deref(args->r0, &fence));
  return iree_hal_fence_signal(fence);
}

IREE_VM_ABI_EXPORT(iree_hal_module_fence_fail,  //
                   iree_hal_module_state_t,     //
                   ri, v) {
  iree_hal_fence_t* fence = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_fence_check_deref(args->r0, &fence));
  iree_status_code_t status_code =
      (iree_status_code_t)(args->i1 & IREE_STATUS_CODE_MASK);
  iree_hal_fence_fail(fence, iree_make_status(status_code));
  return iree_ok_status();
}

// Removes entries in |fences| if they have been reached.
// Returns failure if one or more fences have failed.
static iree_status_t iree_hal_module_fence_elide_reached(
    iree_host_size_t* fence_count, iree_hal_fence_t** fences) {
  iree_host_size_t new_count = *fence_count;
  for (iree_host_size_t i = 0; i < new_count;) {
    iree_status_t status = iree_hal_fence_query(fences[i]);
    if (iree_status_is_ok(status)) {
      // Has been reached; shift the list down.
      memmove(&fences[i], &fences[i + 1],
              (new_count - i - 1) * sizeof(iree_hal_fence_t*));
      fences[new_count - 1] = NULL;
      --new_count;
    } else if (iree_status_is_deferred(status)) {
      // Still waiting.
      iree_status_ignore(status);
      ++i;  // next
    } else {
      // Failed; propagate failure.
      *fence_count = new_count;
      return status;
    }
  }
  *fence_count = new_count;
  return iree_ok_status();
}

// Enters a wait frame for all timepoints in all |fences|.
// Returns an |out_wait_status| of OK if all fences have been reached or
// IREE_STATUS_DEFERRED if one or more fences are still pending and a wait
// frame was entered.
static iree_status_t iree_hal_module_fence_await_begin(
    iree_vm_stack_t* stack, iree_host_size_t fence_count,
    iree_hal_fence_t** fences, iree_timeout_t timeout, iree_zone_id_t zone_id,
    iree_status_t* out_wait_status) {
  // To avoid additional allocations when waiting on multiple fences we enter
  // the wait frame with the maximum required wait source capacity and perform
  // a simple deduplication when building the list. Ideally this helps get us on
  // fast paths of single semaphore waits. The common case is a single fence in
  // which case this is all exceptional.
  iree_host_size_t total_timepoint_capacity = 0;
  for (iree_host_size_t i = 0; i < fence_count; ++i) {
    total_timepoint_capacity += iree_hal_fence_timepoint_count(fences[i]);
  }

  // Fast-path for no semaphores (empty/immediate fences).
  if (total_timepoint_capacity == 0) {
    *out_wait_status = iree_ok_status();
    IREE_TRACE_ZONE_END(zone_id);
    return iree_ok_status();
  }

  // Reserve storage as if all timepoints from all fences were unique.
  iree_vm_wait_frame_t* wait_frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_stack_wait_enter(stack, IREE_VM_WAIT_ALL,
                                                total_timepoint_capacity,
                                                timeout, zone_id, &wait_frame));

  // Insert the first set of timepoints - they're already deduplicated.
  iree_host_size_t unique_timepoint_count = 0;
  if (fence_count >= 1) {
    iree_hal_semaphore_list_t semaphore_list =
        iree_hal_fence_semaphore_list(fences[0]);
    for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
      iree_wait_source_t wait_source = iree_hal_semaphore_await(
          semaphore_list.semaphores[i], semaphore_list.payload_values[i]);
      wait_frame->wait_sources[unique_timepoint_count++] = wait_source;
    }
  }

  // TODO(benvanik): simplify this; it may not be worth the complexity. We'll
  // need more real workloads using multi-fence joins to see how useful this is.

  // Insert remaining fence timepoints by performing merging as we go.
  for (iree_host_size_t i = 1; i < fence_count; ++i) {
    iree_hal_semaphore_list_t semaphore_list =
        iree_hal_fence_semaphore_list(fences[i]);
    for (iree_host_size_t j = 0; j < semaphore_list.count; ++j) {
      // O(n^2) set insertion - relying on this being rare and the total count
      // being low. The savings of a small linear scan here relative to an
      // additional syscall are always worth it but we may want to go further.
      iree_wait_source_t wait_source = iree_hal_semaphore_await(
          semaphore_list.semaphores[j], semaphore_list.payload_values[j]);
      bool found_existing = false;
      for (iree_host_size_t k = 0; k < unique_timepoint_count; ++k) {
        if (wait_frame->wait_sources[k].ctl == wait_source.ctl &&
            wait_frame->wait_sources[k].self == wait_source.self) {
          // Found existing; use max of both.
          wait_frame->wait_sources[k].data =
              iree_max(wait_frame->wait_sources[k].data, wait_source.data);
          found_existing = true;
          break;
        }
      }
      if (!found_existing) {
        wait_frame->wait_sources[unique_timepoint_count++] = wait_source;
      }
    }
  }

  // Update frame with the actual number of timepoints in the wait operation.
  wait_frame->count = unique_timepoint_count;

  *out_wait_status = iree_status_from_code(IREE_STATUS_DEFERRED);
  return iree_ok_status();
}

// PC for iree_hal_module_fence_await.
enum iree_hal_module_fence_await_pc_e {
  // Initial entry point that will try to either wait inline or yield to the
  // scheduler with a wait-all operation.
  IREE_HAL_MODULE_FENCE_AWAIT_PC_BEGIN = 0,
  // Resume entry point after the scheduler wait has resolved (successfully or
  // otherwise).
  IREE_HAL_MODULE_FENCE_AWAIT_PC_RESUME,
};

IREE_VM_ABI_EXPORT(iree_hal_module_fence_await,  //
                   iree_hal_module_state_t,      //
                   iCrD, i) {
  // On entry we either perform the wait or begin a coroutine yield operation.
  // After resuming we check to see if the fence has been reached and propagate
  // the result.
  iree_vm_stack_frame_t* current_frame = iree_vm_stack_top(stack);
  iree_zone_id_t zone_id = 0;
  iree_status_t wait_status = iree_ok_status();
  if (current_frame->pc == IREE_HAL_MODULE_FENCE_AWAIT_PC_BEGIN) {
    uint32_t timeout_millis = (uint32_t)args->i0;
    iree_host_size_t fence_count = 0;
    iree_hal_fence_t** fences = NULL;
    IREE_VM_ABI_VLA_STACK_DEREF(args, a1_count, a1, iree_hal_fence, 32,
                                &fence_count, &fences);

    IREE_TRACE_ZONE_BEGIN(z0);
    zone_id = z0;

    // Capture absolute timeout so that regardless of how long it takes us to
    // wait the user-perceived wait time remains the same.
    iree_timeout_t timeout = timeout_millis == UINT32_MAX
                                 ? iree_infinite_timeout()
                                 : iree_make_timeout_ms(timeout_millis);
    iree_convert_timeout_to_absolute(&timeout);

    // Remove any fences that have been reached and check for failure.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        zone_id, iree_hal_module_fence_elide_reached(&fence_count, fences));

    // If all fences have been reached we can exit early as if we waited
    // successfully.
    if (fence_count > 0) {
      if (iree_all_bits_set(state->flags, IREE_HAL_MODULE_FLAG_SYNCHRONOUS)) {
        // Block the native thread until the fence is reached or the deadline is
        // exceeded.
        for (iree_host_size_t i = 0; i < fence_count; ++i) {
          wait_status = iree_hal_fence_wait(fences[i], timeout);
          if (!iree_status_is_ok(wait_status)) break;
        }
      } else {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            zone_id,
            iree_hal_module_fence_await_begin(stack, fence_count, fences,
                                              timeout, zone_id, &wait_status));
        current_frame->pc = IREE_HAL_MODULE_FENCE_AWAIT_PC_RESUME;
        if (iree_status_is_deferred(wait_status)) {
          zone_id = 0;  // ownership transferred to wait frame
        }
      }
    }
  } else {
    // Resume by leaving the wait frame and storing the result.
    iree_vm_wait_result_t wait_result;
    IREE_RETURN_IF_ERROR(iree_vm_stack_wait_leave(stack, &wait_result));
    wait_status = wait_result.status;
    IREE_TRACE(zone_id = wait_result.trace_zone);
  }

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(wait_status)) {
    // Successful wait.
    rets->i0 = 0;
  } else if (iree_status_is_deferred(wait_status)) {
    // Yielding; resume required.
    // NOTE: zone not ended as it's reserved on the stack.
    status = wait_status;
  } else if (iree_status_is_deadline_exceeded(wait_status)) {
    // Propagate deadline exceeded back to the VM.
    rets->i0 = (int32_t)iree_status_consume_code(wait_status);
    iree_status_ignore(wait_status);
  } else {
    // Fail the invocation.
    status = wait_status;
  }

  IREE_TRACE({
    if (zone_id) IREE_TRACE_ZONE_END(zone_id);
  });
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_create,  //
                   iree_hal_module_state_t,           //
                   rI, r) {
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_check_deref(args->r0, &device));
  uint64_t initial_value = (uint64_t)args->i1;
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(device, initial_value, &semaphore));
  rets->r0 = iree_hal_semaphore_move_ref(semaphore);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_query,  //
                   iree_hal_module_state_t,          //
                   r, iI) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));
  uint64_t current_value = 0;
  iree_status_t query_status =
      iree_hal_semaphore_query(semaphore, &current_value);
  rets->i0 = iree_status_consume_code(query_status);
  rets->i1 = current_value;
  iree_status_ignore(query_status);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_signal,  //
                   iree_hal_module_state_t,           //
                   rI, v) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));
  uint64_t new_value = (uint64_t)args->i1;
  return iree_hal_semaphore_signal(semaphore, new_value);
}

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_fail,  //
                   iree_hal_module_state_t,         //
                   ri, v) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));
  iree_status_code_t status_code =
      (iree_status_code_t)(args->i1 & IREE_STATUS_CODE_MASK);
  iree_hal_semaphore_fail(semaphore, iree_make_status(status_code));
  return iree_ok_status();
}

// PC for iree_hal_module_semaphore_await.
enum iree_hal_module_semaphore_await_pc_e {
  // Initial entry point that will try to either wait inline or yield to the
  // scheduler with a wait-all operation.
  IREE_HAL_MODULE_SEMAPHORE_AWAIT_PC_BEGIN = 0,
  // Resume entry point after the scheduler wait has resolved (successfully or
  // otherwise).
  IREE_HAL_MODULE_SEMAPHORE_AWAIT_PC_RESUME,
};

IREE_VM_ABI_EXPORT(iree_hal_module_semaphore_await,  //
                   iree_hal_module_state_t,          //
                   rI, i) {
  // On entry we either perform the wait or begin a coroutine yield operation.
  // After resuming we check to see if the timepoint has been reached and
  // propagate the result.
  iree_vm_stack_frame_t* current_frame = iree_vm_stack_top(stack);
  iree_zone_id_t zone_id = 0;
  iree_status_t wait_status = iree_ok_status();
  if (current_frame->pc == IREE_HAL_MODULE_SEMAPHORE_AWAIT_PC_BEGIN) {
    iree_hal_semaphore_t* semaphore = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_check_deref(args->r0, &semaphore));
    uint64_t new_value = (uint64_t)args->i1;

    IREE_TRACE_ZONE_BEGIN(z0);
    zone_id = z0;

    // TODO(benvanik): take timeout as an argument.
    // Capture absolute timeout so that regardless of how long it takes us to
    // wait the user-perceived wait time remains the same.
    iree_timeout_t timeout = iree_infinite_timeout();
    iree_convert_timeout_to_absolute(&timeout);

    if (iree_all_bits_set(state->flags, IREE_HAL_MODULE_FLAG_SYNCHRONOUS)) {
      // Block the native thread until the fence is reached or the deadline is
      // exceeded.
      wait_status = iree_hal_semaphore_wait(semaphore, new_value, timeout);
    } else {
      // Quick check inline before yielding to the scheduler. This avoids a
      // round-trip through the scheduling stack for cases where we complete
      // synchronously.
      //
      // The query may fail to indicate that the semaphore is in a failure
      // state and we propagate the failure status to the waiter.
      //
      // It's possible to race here if we get back an older value and then
      // before we wait the target is reached but that's ok: the wait will
      // always be correctly ordered.
      uint64_t current_value = 0ull;
      wait_status = iree_hal_semaphore_query(semaphore, &current_value);
      if (iree_status_is_ok(wait_status) && current_value < new_value) {
        // Enter a wait frame and yield execution back to the scheduler.
        // When the wait handle resolves we'll resume at the RESUME PC.
        iree_vm_wait_frame_t* wait_frame = NULL;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            zone_id, iree_vm_stack_wait_enter(stack, IREE_VM_WAIT_ALL, 1,
                                              timeout, zone_id, &wait_frame));
        wait_frame->wait_sources[0] =
            iree_hal_semaphore_await(semaphore, new_value);
        current_frame->pc = IREE_HAL_MODULE_SEMAPHORE_AWAIT_PC_RESUME;
        wait_status = iree_status_from_code(IREE_STATUS_DEFERRED);
        zone_id = 0;  // ownership transferred to wait frame
      }
    }
  } else {
    // Resume by leaving the wait frame and storing the result.
    iree_vm_wait_result_t wait_result;
    IREE_RETURN_IF_ERROR(iree_vm_stack_wait_leave(stack, &wait_result));
    wait_status = wait_result.status;
    IREE_TRACE(zone_id = wait_result.trace_zone);
  }

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(wait_status)) {
    // Successful wait.
    rets->i0 = 0;
  } else if (iree_status_is_deferred(wait_status)) {
    // Yielding; resume required.
    // NOTE: zone not ended as it's reserved on the stack.
    status = wait_status;
  } else if (iree_status_is_deadline_exceeded(wait_status)) {
    // Propagate deadline exceeded back to the VM.
    rets->i0 = (int32_t)iree_status_consume_code(wait_status);
    iree_status_ignore(wait_status);
  } else {
    // Fail the invocation.
    status = wait_status;
  }

  IREE_TRACE({
    if (zone_id) IREE_TRACE_ZONE_END(zone_id);
  });
  return status;
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_hal_module_exports_ table.
static const iree_vm_native_function_ptr_t iree_hal_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "iree/modules/hal/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t iree_hal_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_hal_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .attr_count = 0,                                             \
      .attrs = NULL,                                               \
  },
#include "iree/modules/hal/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_hal_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_hal_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t iree_hal_module_descriptor_ = {
    .module_name = iree_string_view_literal("hal"),
    .module_attr_count = 0,
    .module_attrs = NULL,
    .import_count = 0,  // workaround for 0-length C struct
    .imports = iree_hal_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_hal_module_exports_),
    .exports = iree_hal_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_hal_module_funcs_),
    .functions = iree_hal_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_hal_module_create(
    iree_hal_device_t* device, iree_hal_module_flags_t flags,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_hal_module_destroy,
      .alloc_state = iree_hal_module_alloc_state,
      .free_state = iree_hal_module_free_state,
      .notify = iree_hal_module_notify,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_hal_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_hal_module_descriptor_, host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_hal_module_t* module = IREE_HAL_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;
  module->flags = flags;
  module->shared_device = device;
  iree_hal_device_retain(module->shared_device);

  *out_module = base_module;
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_device_t* iree_hal_module_state_device(
    iree_vm_module_state_t* module_state) {
  iree_hal_module_state_t* state = (iree_hal_module_state_t*)module_state;
  return state->shared_device;
}

//===--------------------------------------------------------------------===//
// Utilities
//===--------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_assign(
    const iree_vm_list_t* list, iree_host_size_t i) {
  return (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
      list, i, iree_hal_buffer_view_get_descriptor());
}

IREE_API_EXPORT iree_hal_buffer_view_t* iree_vm_list_get_buffer_view_retain(
    const iree_vm_list_t* list, iree_host_size_t i) {
  iree_hal_buffer_view_t* value = iree_vm_list_get_buffer_view_assign(list, i);
  iree_hal_buffer_view_retain(value);
  return value;
}

IREE_API_EXPORT iree_status_t iree_vm_list_set_buffer_view_retain(
    iree_vm_list_t* list, iree_host_size_t i, iree_hal_buffer_view_t* value) {
  iree_vm_ref_t value_ref;
  IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
      value, iree_hal_buffer_view_type_id(), &value_ref));
  return iree_vm_list_set_ref_retain(list, i, &value_ref);
}
