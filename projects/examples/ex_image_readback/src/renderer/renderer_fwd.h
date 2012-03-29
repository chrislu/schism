
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LARGE_DATA_RENDERER_H_INCLUDED
#define SCM_LARGE_DATA_RENDERER_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace data {

class readback_benchmark;

typedef shared_ptr<readback_benchmark>          readback_benchmark_ptr;
typedef shared_ptr<readback_benchmark const>    readback_benchmark_cptr;

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_RENDERER_H_INCLUDED
