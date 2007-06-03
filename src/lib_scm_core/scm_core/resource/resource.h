
#ifndef RESOURCE_H_INCLUDED
#define RESOURCE_H_INCLUDED


#include <boost/weak_ptr.hpp>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace res {

class resource_manager_base;

template<class res_type>
class resource_manager;

class __scm_export resource_base
{
public:
    resource_base();
    resource_base(const resource_base& /*res*/);
    virtual ~resource_base();

    virtual const resource_base&    operator=(const resource_base& /*rhs*/) = 0;

                                    operator bool() const;
    bool                            operator !() const;

    virtual                         swap(resource_base& /*ref*/)    = 0;

protected:

private:
    boost::weak_ptr<resource_manager_base>  _manager;

    void                            register_instance();
    void                            release_instance();

}; // class resource_base



template<class res_type>
class resource : public resource_base
{
public:
    resource();
    resource(const resource& /*res*/);
    virtual ~resource();

    const resource_base&            operator=(const resource_base& /*rhs*/);

                                    operator bool() const;
    bool                            operator !() const;

    res_type&                       get() const;
    virtual void                    swap(resource_base& /*ref*/);

private:
    explicit resource(const boost::weak_ptr<res_type>& res,
                      const boost::weak_ptr<resource_manager<res_type> >& mgr);

    void                            register_instance();
    void                            release_instance();

    boost::weak_ptr<res_type>                       _resource;
    boost::weak_ptr<resource_manager<res_type> >    _manager;

    friend class resource_manager<res_type>;

}; // class resource

} // namespace res
} // namespace scm

#include "resource.inl"

#include <scm_core/utilities/platform_warning_enable.h>

#endif // RESOURCE_H_INCLUDED
