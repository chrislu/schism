
#ifndef RESOURCE_TYPE_BASE_H_INCLUDED
#define RESOURCE_TYPE_BASE_H_INCLUDED


#include <boost/weak_ptr.hpp>

namespace scm {
namespace res {

template <class desc_type>
class resource_type_base
{
public:
    typedef desc_type   descriptor_type;

public:
    resource_type_base() {}
    resource_type_base(const descriptor_type& desc) : _descriptor(desc) {}

    virtual const descriptor_type&  get_descriptor() const { return (_descriptor); }

private:
    descriptor_type     _descriptor;

}; // struct resource_type_base

} // namespace res
} // namespace scm

#endif // RESOURCE_TYPE_BASE_H_INCLUDED
