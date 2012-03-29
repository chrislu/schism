
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LARGE_DATA_RENDERER_FWD_H_INCLUDED
#define SCM_LARGE_DATA_RENDERER_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace data {

class volume_data;
class volume_renderer;

typedef shared_ptr<volume_data>             volume_data_ptr;
typedef shared_ptr<volume_data const>       volume_data_cptr;

typedef shared_ptr<volume_renderer>         volume_renderer_ptr;
typedef shared_ptr<volume_renderer const>   volume_renderer_cptr;

class cuda_volume_data;
class cuda_volume_renderer;

typedef shared_ptr<cuda_volume_data>             cuda_volume_data_ptr;
typedef shared_ptr<cuda_volume_data const>       cuda_volume_data_cptr;

typedef shared_ptr<cuda_volume_renderer>         cuda_volume_renderer_ptr;
typedef shared_ptr<cuda_volume_renderer const>   cuda_volume_renderer_cptr;

} // namespace data
} // namespace scm

#endif // SCM_LARGE_DATA_RENDERER_FWD_H_INCLUDED
