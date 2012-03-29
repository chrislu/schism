
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <cassert>

namespace std
{
    template<class res_type>
    inline void swap(scm::res::resource_pointer<res_type>& lhs,
                     scm::res::resource_pointer<res_type>& rhs)
    {
        lhs.swap(rhs);
    }
} // namespace std

namespace scm {
namespace res {

template<class res_type>
resource_pointer<res_type>::resource_pointer()
{
}

template<class res_type>
resource_pointer<res_type>::resource_pointer(const resource_pointer<res_type>& res)
  : resource_pointer_base(res)
{
}

template<class res_type>
resource_pointer<res_type>::~resource_pointer()
{
}

template<class res_type>
resource_pointer<res_type>::resource_pointer(const typename resource_pointer<res_type>::resource_ptr&  res,
                                             const typename resource_pointer<res_type>::manager_ptr&   mgr)
  : resource_pointer_base(res, mgr)
{
}

template<class res_type>
inline res_type&
resource_pointer<res_type>::get()
{
    return (dynamic_cast<resource_manager<res_type>&>(*_manager.lock()).to_resource(*this));
}

template<class res_type>
inline const res_type&
resource_pointer<res_type>::get() const
{
    return (dynamic_cast<const resource_manager<res_type>&>(*_manager.lock()).to_resource(*this));
}

} // namespace res
} // namespace scm
