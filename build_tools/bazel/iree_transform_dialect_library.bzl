# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling transform dialect libraries to MLIR bytecode"""

load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

def iree_transform_dialect_library(
        name,
        src,
        module_name,
        flags,
        deps = [],
        **kwargs):
    """Compiles a transform dialect library into MLIR bytecode.

    Args:
        name: Name of the target
        src: mlir source file to be compiled.
        module_name: Optional name for the generated IREE module.
            Defaults to `name.mlirbc`.
        flags: additional flags to pass to the compiler.
        deps: Optional. Dependencies to add to the generated library.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    if not module_name:
        module_name = "%s.mlirbc" % (name)

    out_files = [module_name]
    compile_tool = "//tools:iree-opt"

    native.genrule(
        name = name,
        srcs = [src],
        outs = out_files,
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (compile_tool),
                " ".join(flags),
                "-o $(location %s)" % (module_name),
                "$(location %s)" % (src),
            ]),
        ]),
        tools = [compile_tool],
        message = "Compiling transform dialect library %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )
