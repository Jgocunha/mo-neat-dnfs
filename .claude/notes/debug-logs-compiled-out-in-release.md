# DEBUG log lines do not exist in Release builds

`neat_dnfs::tools::logger::log()` (src/neat_tools/logger.cpp) returns immediately for
`LogLevel::DEBUG` unless `_DEBUG` is defined, so a Release binary (the only one the evol apps
and the test lane normally use) never prints a DEBUG line, whatever
`Logger::setMinLogLevel()` says. `setMinLogLevel(DEBUG)` only matters in a Debug build.

To read a DEBUG line on a real run (for example Population's per-generation Pareto sentence,
"gen 7: 5 fronts, front 0 holds ..."), either:

- build Debug (`scripts/build.bat` builds it by default; the DNF simulation is much slower), or
- raise that one call to `INFO` in a scratch Release build, run, read, and revert it before
  committing.
