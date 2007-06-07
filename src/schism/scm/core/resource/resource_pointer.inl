
#include <cassert>

//namespace std
//{
//    template<class res_type>
//    inline void swap(scm::res::resource<res_type>& lhs,
//                     scm::res::resource<res_type>& rhs)
//    {
//        lhs.swap(rhs);
//    }
//} // namespace std

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
inline res_type&
resource_pointer<res_type>::get()
{
    return (dynamic_cast<resource_manager<res_type>&>(*_manager.lock()).to_resource(*this));
}

} // namespace res
} // namespace scm
