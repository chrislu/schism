
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <scm/core/resource/resource_pointer.h>

namespace scm {
namespace res {

template<class res_type>
resource_manager<res_type>::resource_manager()
{
}

template<class res_type>
resource_manager<res_type>::~resource_manager()
{
}

template<class res_type>
bool
resource_manager<res_type>::is_loaded(const resource_descriptor_type& desc) const
{
    return (resource_manager_base::is_loaded(desc.hash_value()));
}

template<class res_type>
resource_pointer<res_type>
resource_manager<res_type>::retrieve_instance(const resource_descriptor_type& desc)
{
    return (to_resource_ptr(retrieve_instance(desc.hash_value())));
}

template<class res_type>
resource_pointer<res_type>
resource_manager<res_type>::retrieve_instance(const resource_base::hash_type hash)
{
    resource_container::const_iterator    res_it = _resources.find(hash);

    if (res_it != _resources.end()) {
        return(resource_pointer<res_type>(res_it->second.first, shared_from_this()));
    }
    else {
        return(resource_pointer<res_type>());
    }
}

template<class res_type>
resource_pointer<res_type>
resource_manager<res_type>::create_instance(const resource_descriptor_type& desc)
{
    if (is_loaded(desc)) {
        return (retrieve_instance(desc));
    }
    else {
        res_ptr_type ptr(new res_type(desc));
        insert_instance_values(desc.hash_value(),
                               ptr); // ptr inst copied into manager

        return (resource_pointer<res_type>(ptr, shared_from_this()));
    }
}

template<class res_type>
res_type&
resource_manager<res_type>::to_resource(resource_pointer_base& ref) const
{
    return (dynamic_cast<res_type&>(*ref._resource.lock()));
}

template<class res_type>
const res_type&
resource_manager<res_type>::to_resource(const resource_pointer_base& ref) const
{
    return (dynamic_cast<const res_type&>(*ref._resource.lock()));
}

template<class res_type>
resource_pointer<res_type>&
resource_manager<res_type>::to_resource_ptr(resource_pointer_base& ref) const
{
    //return (static_cast<resource_pointer<res_type>&>(ref));
    return (dynamic_cast<resource_pointer<res_type>&>(ref));
}

template<class res_type>
const resource_pointer<res_type>&
resource_manager<res_type>::to_resource_ptr(const resource_pointer_base& ref) const
{
    //return (static_cast<resource_pointer<res_type>&>(ref));
    return (dynamic_cast<const resource_pointer<res_type>&>(ref));
}

} // namespace res
} // namespace scm
