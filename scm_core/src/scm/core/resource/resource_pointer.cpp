
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#if 0

#include "resource_pointer.h"

#include <cassert>

#include <scm/core/resource/resource.h>
#include <scm/core/resource/resource_manager.h>

namespace std
{
    inline void swap(scm::res::resource_pointer_base& lhs,
                     scm::res::resource_pointer_base& rhs)
    {
        lhs.swap(rhs);
    }
} // namespace std


namespace scm {
namespace res {

resource_pointer_base::resource_pointer_base()
{
}

resource_pointer_base::resource_pointer_base(const resource_pointer_base& res)
  : _resource(res._resource),
    _manager(res._manager)
{
    register_instance();
}

resource_pointer_base::resource_pointer_base(const resource_pointer_base::resource_ptr&  res,
                                             const resource_pointer_base::manager_ptr&   mgr)
  : _resource(res),
    _manager(mgr)
{
    register_instance();
}

resource_pointer_base::~resource_pointer_base()
{
    release_instance();
}

const resource_pointer_base& resource_pointer_base::operator=(const resource_pointer_base& rhs)
{
    release_instance();

    _manager    = rhs._manager;
    _resource   = rhs._resource;

    register_instance();

    return (*this);
}

bool resource_pointer_base::operator==(const resource_pointer_base& rhs) const
{
    boost::shared_ptr<resource_base> l = _resource.lock();
    boost::shared_ptr<resource_base> r = rhs._resource.lock();
    
    if (l && r) {
        return (*l == *r);
    }

    return (false);
}

resource_pointer_base::operator bool() const
{
    return (_resource.lock());
}

bool resource_pointer_base::operator !() const
{
    return (!_resource.lock());
}

void resource_pointer_base::swap(resource_pointer_base& ref)
{
    std::swap(_manager,  ref._manager);
    std::swap(_resource, ref._resource);
}

void resource_pointer_base::register_instance()
{
    assert(!_resource.lock() == !_manager.lock());

    if (boost::shared_ptr<resource_manager_base> m = _manager.lock()) {
        m->register_instance(*this);
    }
}

void resource_pointer_base::release_instance()
{
    assert(!_resource.lock() == !_manager.lock());

    if (boost::shared_ptr<resource_manager_base> m = _manager.lock()) {
        m->release_instance(*this);
    }
}

} // namespace res
} // namespace scm

#endif // 0
