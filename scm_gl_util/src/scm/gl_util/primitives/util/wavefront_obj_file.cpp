
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "wavefront_obj_file.h"

#include <boost/lexical_cast.hpp>

namespace scm {
namespace gl {
namespace util {

wavefront_material::wavefront_material()
  : _Ns(128.0f),
    _Ni(1.0),
    _Ka(0.125f, 0.125f, 0.125f, 1.0f),
    _Kd(0.875f, 0.625f, 0.5f,   1.0f),
    _Ks(0.25f,  0.25f,  0.25f,  1.0f),
    _Tf(1.f,    1.f,    1.f,    1.0f),
    _d(1.0f)
{
}


wavefront_model::object_container::iterator wavefront_model::add_new_object()
{
    std::string     object_name =   std::string("object_") 
                                  + boost::lexical_cast<std::string>(_objects.size() - 1);

    return (add_new_object(object_name));
}

wavefront_model::object_container::iterator wavefront_model::add_new_object(const std::string& name)
{
    object_container::iterator  new_obj;

    _objects.push_back(wavefront_object());

    new_obj = _objects.end() - 1;
    new_obj->_name = name;

    return (new_obj);
}

wavefront_object::group_container::iterator wavefront_object::add_new_group()
{
    std::string     group_name  =   std::string("group_") 
                                  + boost::lexical_cast<std::string>(_groups.size() - 1);

    return (add_new_group(group_name));
}

wavefront_object::group_container::iterator wavefront_object::add_new_group(const std::string& name)
{
    group_container::iterator  new_grp;

    _groups.push_back(wavefront_object_group());

    new_grp = _groups.end() - 1;
    new_grp->_name = name;

    return (new_grp);
}

} // namespace util
} // namespace gl
} // namespace scm
