
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
resource<res_type>::resource()
{
}

template<class res_type>
resource<res_type>::resource(const resource<res_type>& res)
  : resource_base(res)
{
}

template<class res_type>
resource<res_type>::resource(const resource_base::resource_ptr&  res,
                             const resource_base::manager_ptr&   mgr)
  : resource_base(res, mgr)
{
}

template<class res_type>
resource<res_type>::~resource()
{
}

/*
template<class res_type>
resource<res_type>::resource(const resource<res_type>& res)
{
    *this = res;
}

template<class res_type>
resource<res_type>::resource(const boost::weak_ptr<res_type>& res,
                             const boost::weak_ptr<resource_manager<res_type> >& mgr)
  : _resource(res),
    _manager(mgr)
{
}

template<class res_type>
resource<res_type>::~resource()
{
    release_instance();
}
 

template<class res_type>
const resource<res_type>&
resource<res_type>::operator=(const resource<res_type>& rhs)
{
    release_instance();

    _manager    = rhs._manager;
    _resource   = rhs._resource;

    register_instance();

    return (*this);
}

template<class res_type>
inline resource<res_type>::operator bool() const
{
    return (_resource.lock());
}
template<class res_type>
inline bool resource<res_type>::operator !() const
{
    return (!_resource.lock());
}
*/

template<class res_type>
inline res_type&
resource<res_type>::get() const
{
    return (*_resource.lock());
}
/*
template<class res_type>
void resource<res_type>::register_instance()
{
    if (boost::shared_ptr<resource_manager<res_type> > m = _manager.lock()) {
        m->register_instance(*this);
    }
}

template<class res_type>
void resource<res_type>::release_instance()
{
    if (boost::shared_ptr<resource_manager<res_type> > m = _manager.lock()) {
        m->release_instance(*this);
    }
}

template<class res_type>
void resource<res_type>::swap(resource<res_type>& ref)
{
    std::swap(_manager,  ref._manager);
    std::swap(_resource, ref._resource);
}
*/
} // namespace res
} // namespace scm
