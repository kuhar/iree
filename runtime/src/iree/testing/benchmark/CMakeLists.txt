# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#===------------------------------------------------------------------------===#
# Strategy libraries (pure state machines, no I/O)
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    strategy
  HDRS
    "strategy.h"
    "strategy_auto.h"
    "strategy_fixed.h"
    "strategy_time_budget.h"
  SRCS
    "strategy_auto.c"
    "strategy_fixed.c"
    "strategy_time_budget.c"
  DEPS
    iree::base
  PUBLIC
)

#===------------------------------------------------------------------------===#
# Timer abstraction
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    timer
  HDRS
    "timer.h"
  SRCS
    "timer.c"
  DEPS
    iree::base
  PUBLIC
)

#===------------------------------------------------------------------------===#
# Statistics (pure functions)
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    stats
  HDRS
    "stats.h"
  SRCS
    "stats.c"
  PUBLIC
)

#===------------------------------------------------------------------------===#
# Config, Result, Registry
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    config
  HDRS
    "config.h"
    "result.h"
  DEPS
    ::strategy
    iree::base
  PUBLIC
)

iree_cc_library(
  NAME
    registry
  HDRS
    "registry.h"
  SRCS
    "registry.c"
  DEPS
    ::config
  PUBLIC
)

#===------------------------------------------------------------------------===#
# State (iteration loop)
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    state
  HDRS
    "state.h"
  SRCS
    "state.c"
  DEPS
    ::config
    ::strategy
    ::timer
  PUBLIC
)

#===------------------------------------------------------------------------===#
# Reporter interface + implementations
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    reporter
  HDRS
    "reporter.h"
  SRCS
    "reporter_console.c"
    "reporter_json.c"
  DEPS
    ::config
  PUBLIC
)

#===------------------------------------------------------------------------===#
# Runner (orchestrator)
#===------------------------------------------------------------------------===#

iree_cc_library(
  NAME
    runner
  HDRS
    "runner.h"
  SRCS
    "runner.c"
  DEPS
    ::config
    ::registry
    ::reporter
    ::state
    ::stats
    ::strategy
    ::timer
    iree::base
  PUBLIC
)

#===------------------------------------------------------------------------===#
# Tests
#===------------------------------------------------------------------------===#

iree_cc_test(
  NAME
    strategy_test
  SRCS
    "strategy_test.cc"
  DEPS
    ::config
    ::strategy
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    stats_test
  SRCS
    "stats_test.cc"
  DEPS
    ::stats
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    registry_test
  SRCS
    "registry_test.cc"
  DEPS
    ::config
    ::registry
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    runner_test
  SRCS
    "runner_test.cc"
  DEPS
    ::config
    ::registry
    ::reporter
    ::runner
    ::state
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
)

iree_cc_test(
  NAME
    reporter_test
  SRCS
    "reporter_test.cc"
  DEPS
    ::config
    ::reporter
    ::strategy
    iree::testing::gtest
    iree::testing::gtest_main
)
