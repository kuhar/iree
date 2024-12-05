# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_transform_dialect_library()
#
# Compiles a transform dialect library to MLIR bytecode.
#
# Parameters:
# NAME: Name of target (see Note).
# SRC: Source file to compile into a bytecode module.
# FLAGS: Flags to pass to the compiler tool (list of strings).
#     `--output-format=vm-bytecode` is included automatically.
# MODULE_FILE_NAME: Optional. When specified, sets the output bytecode module
#    file name. When not specified, a default file name will be generated from
#    ${NAME}.
# FRIENDLY_NAME: Optional. Name to use to display build progress info.
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
#     Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC
#     will be in ${PACKAGE}/internal.
# TESTONLY: When added, this target will only be built if IREE_BUILD_TESTS=ON.
# DEPENDS: Optional. Additional dependencies beyond SRC and the tools.
#
# Note:
# By default, iree_transform_dialect_library will create a module target named
# ${NAME}.mlirb. The iree:: form should always be used to reduce namespace pollution.
function(iree_transform_dialect_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;MODULE_FILE_NAME;FRIENDLY_NAME"
    "FLAGS;DEPENDS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  set(_COMPILE_TOOL "iree-opt")

  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else()
    set(_MODULE_FILE_NAME "${_RULE_NAME}.mlirbc")
  endif()

  set(_ARGS "")
  list(APPEND _ARGS "${_RULE_FLAGS}")

  get_filename_component(_SRC_PATH "${_RULE_SRC}" REALPATH)
  list(APPEND _ARGS "${_SRC_PATH}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_MODULE_FILE_NAME}")

  set(_OUTPUT_FILES "${_MODULE_FILE_NAME}")


  if(_RULE_FRIENDLY_NAME)
    set(_FRIENDLY_NAME "${_RULE_FRIENDLY_NAME}")
  else()
    get_filename_component(_FRIENDLY_NAME "${_RULE_SRC}" NAME)
  endif()

  set(_DEPENDS "")
  iree_package_ns(_PACKAGE_NAME)
  list(TRANSFORM _RULE_DEPENDS REPLACE "^::" "${_PACKAGE_NAME}::")
  foreach(_DEPEND ${_RULE_DEPENDS})
    string(REPLACE "::" "_" _DEPEND "${_DEPEND}")
    list(APPEND _DEPENDS ${_DEPEND})
  endforeach()

  add_custom_command(
    OUTPUT
      ${_OUTPUT_FILES}
    COMMAND
      ${_COMPILE_TOOL}
      ${_ARGS}
    DEPENDS
      ${_COMPILE_TOOL}
      ${_RULE_SRC}
      ${_DEPENDS}
    COMMENT
      "Generating ${_MODULE_FILE_NAME} from ${_FRIENDLY_NAME}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_MODULE_FILE_NAME}"
  )

  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG "TESTONLY")
  endif()
  if(_RULE_PUBLIC)
    set(_PUBLIC_ARG "PUBLIC")
  endif()

endfunction()
