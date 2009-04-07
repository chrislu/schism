
#include "scm_geom.h"

namespace scm {
namespace data {

geometry_descriptor::geometry_descriptor()
  : _version(1),
    _geometry_origin(0.f, 0.f, 0.f),
    _geometry_scale(1.f, 1.f, 1.f)
{
}

geometry_descriptor::~geometry_descriptor()
{
}

geometry_descriptor::geometry_descriptor(const geometry_descriptor& rhs)
  : _version(rhs._version),
    _geometry_origin(rhs._geometry_origin),
    _geometry_scale(rhs._geometry_scale),
    _sobj_file(rhs._sobj_file),
    _name(rhs._name)
{
}

const geometry_descriptor& geometry_descriptor::operator=(const geometry_descriptor&  rhs)
{
    _geometry_origin    = rhs._geometry_origin;
    _geometry_scale     = rhs._geometry_scale;
    _sobj_file          = rhs._sobj_file;
    _name               = rhs._name;

    return (*this);
}


std::ostream& operator<<(std::ostream& os, const geometry_descriptor& desc)
{
    os << "scm_geom " << desc._version << std::endl
       << desc._geometry_origin.x << " " << desc._geometry_origin.y << " " << desc._geometry_origin.z << std::endl
       << desc._geometry_scale.x  << " " << desc._geometry_scale.y  << " " << desc._geometry_scale.z << std::endl
       << desc._sobj_file << std::endl
       << desc._name << std::endl;
    
    return (os);
}

std::istream& operator>>(std::istream& i, geometry_descriptor& desc)
{
    std::string head;

    i >> head;
    if (!i)  return (i);

    if (head != "scm_geom") {
        i.setstate(std::ios_base::failbit);
        return (i);
    }

    unsigned v;

    i >> v;
    if (!i)  return (i);

    if (v != desc._version) {
        i.setstate(std::ios_base::failbit);
        return (i);
    }

    i >> desc._geometry_origin.x; if (!i)  return (i);
    i >> desc._geometry_origin.y; if (!i)  return (i);
    i >> desc._geometry_origin.z; if (!i)  return (i);

    i >> desc._geometry_scale.x; if (!i)  return (i);
    i >> desc._geometry_scale.y; if (!i)  return (i);
    i >> desc._geometry_scale.z; if (!i)  return (i);

    i.ignore(); // newline ignore

    std::getline(i, desc._sobj_file); if (!i)  return (i);

    std::getline(i, desc._name); if (!i)  return (i);

    i.setstate(std::ios_base::goodbit);

    return (i);
}

} // namespace data
} // namespace scm
