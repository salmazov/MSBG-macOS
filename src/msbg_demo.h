/******************************************************************************
 *
 * Copyright 2025 Bernhard Braun 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 ******************************************************************************/

#ifndef MSBG_TEST_H
#define MSBG_TEST_H

#ifndef VIS_OUTPUT_DIR
#define VIS_OUTPUT_DIR "out_msbg_demo"
#endif

enum BackendType {
  BACKEND_CPU = 0,
  BACKEND_GPU = 1
};

extern float camPos[3];
extern float camLookAt[3];
extern float camLight[3];
extern float camZoom;
extern int camRes[2];

int msbg_test_sparse(int testCase, const char *basePointsFile, 
    		     int bsx0, int sx, int sy, int sz,
                     BackendType backend,
                     bool validateAgainstCpu=false);


#endif // MSBG_TEST_H
