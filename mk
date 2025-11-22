#!/bin/bash

CC_BIN=${CC_BIN:-/opt/homebrew/bin/gcc-15}
CXX_BIN=${CXX_BIN:-/opt/homebrew/bin/g++-15}

make -f ../makefile \
  OBJE=o\
  CPP_FLAGS="-std=gnu++17"\
  CFLAGS_OPT="-O3 -DNDEBUG -mtune=native" \
  CFLAGS_PROF= \
  CFLAGS_OMP=-fopenmp \
  CFLAGS_TBB="" \
  LIB_OMP="-lgomp"\
  CC="${CC_BIN}" \
  LD="${CXX_BIN}  -o" \
  LDFLAGS_BW=\
  AR="ar" \
  ISPC="ispc"\
  CFLAGS_BW="-m64 -DMI_WITH_64BIT"\
  -j \
  $1
