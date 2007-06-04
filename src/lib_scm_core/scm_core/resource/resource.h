
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
protected:
    typedef boost::weak_ptr<resource_base>          resource_ptr;
    typedef boost::weak_ptr<resource_manager_base>  manager_ptr;

public:
    resource_base();
    resource_base(const resource_base& /*res*/);
    virtual ~resource_base();

    virtual const resource_base&    operator=(const resource_base& /*rhs*/);

                                    operator bool() const;
    bool                            operator !() const;

    void                            swap(resource_base& /*ref*/);

protected:
    explicit resource_base(const resource_ptr&  /*res*/,
                           const manager_ptr&   /*mgr*/);

private:
    resource_ptr                    _resource;
    manager_ptr                     _manager;

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

    res_type&                       get() const;

private:
    explicit resource(const resource_ptr&  /*res*/,
                      const manager_ptr&   /*mgr*/);

    friend class resource_manager<res_type>;

}; // class resource

} // namespace res
} // namespace scm

#include "resource.inl"

#include <scm_core/utilities/platform_warning_enable.h>

#endif // RESOURCE_H_INCLUDED
