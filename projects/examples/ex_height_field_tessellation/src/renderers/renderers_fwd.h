
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_LDATA_RENDERERS_H_INCLUDED
#define SCM_LDATA_RENDERERS_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace data {

class patch_grid_mesh;
class height_field_data;
class height_field_tessellator;

typedef shared_ptr<patch_grid_mesh>                 patch_grid_mesh_ptr;
typedef shared_ptr<patch_grid_mesh const>           patch_grid_mesh_cptr;

typedef shared_ptr<height_field_data>               height_field_data_ptr;
typedef shared_ptr<height_field_data const>         height_field_data_cptr;

typedef shared_ptr<height_field_tessellator>        height_field_tessellator_ptr;
typedef shared_ptr<height_field_tessellator const>  height_field_tessellator_cptr;

} // namespace data
} // namespace scm

#endif // SCM_LDATA_RENDERERS_H_INCLUDED
