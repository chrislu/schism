
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef RESOURCE_POINTER_H_INCLUDED
#define RESOURCE_POINTER_H_INCLUDED

#if 0

#include <cstddef>

#include <scm/core/memory.h>
#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace res {

class resource_base;
class resource_manager_base;

template<class res_type>
class resource_manager;

class __scm_export(core) resource_pointer_base
{
protected:
    typedef weak_ptr<resource_base>             resource_ptr;
    typedef weak_ptr<resource_manager_base>     manager_ptr;

public:
    resource_pointer_base();
    resource_pointer_base(const resource_pointer_base& /*res*/);
    virtual ~resource_pointer_base();

    const resource_pointer_base&    operator=(const resource_pointer_base&  /*rhs*/);
    bool                            operator==(const resource_pointer_base& /*rhs*/) const;

                                    operator bool() const;
    bool                            operator !() const;

    void                            swap(resource_pointer_base& /*ref*/);

protected:
    explicit resource_pointer_base(const resource_ptr&  /*res*/,
                                   const manager_ptr&   /*mgr*/);

    resource_ptr                    _resource;
    manager_ptr                     _manager;

    void                            register_instance();
    void                            release_instance();

private:
    friend class resource_manager_base;
    template<class res_type> friend class resource_manager;

}; // class resource_pointer_base

template<class res_type>
class resource_pointer : public resource_pointer_base
{
public:
    resource_pointer();
    resource_pointer(const resource_pointer& /*res*/);
    virtual ~resource_pointer();

    res_type&                       get();
    const res_type&                 get() const;

private:
    explicit resource_pointer(const resource_ptr&  /*res*/,
                              const manager_ptr&   /*mgr*/);
    template<class res_type_> friend class resource_manager;

}; // class resource_pointer

} // namespace res
} // namespace scm

#include "resource_pointer.inl"

#include <scm/core/utilities/platform_warning_enable.h>

#endif // 0

#endif // RESOURCE_POINTER_H_INCLUDED
