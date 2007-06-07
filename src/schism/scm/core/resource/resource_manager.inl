
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
    return (to_resource_ptr(resource_manager_base::retrieve_instance(desc.hash_value())));
}

template<class res_type>
resource_pointer<res_type>
resource_manager<res_type>::create_instance(const resource_descriptor_type& desc)
{
    if (is_loaded(desc)) {
        return (retrieve_instance(desc));
    }
    else {
        return (to_resource_ptr(insert_instance(desc.hash_value(),
                                                res_ptr_type(new res_type(desc)))));
    }
}

template<class res_type>
res_type&
resource_manager<res_type>::to_resource(resource_pointer_base& ref) const
{
    return (dynamic_cast<res_type&>(*ref._resource.lock()));
}

template<class res_type>
resource_pointer<res_type>&
resource_manager<res_type>::to_resource_ptr(resource_pointer_base& ref) const
{
    return (dynamic_cast<resource_pointer<res_type>&>(ref));
}

} // namespace res
} // namespace scm
