
#include "resource.h"

#include <cassert>

#include <scm_core/resource/resource_manager.h>

namespace std
{
    inline void swap(scm::res::resource_base& lhs,
                     scm::res::resource_base& rhs)
    {
        lhs.swap(rhs);
    }
} // namespace std


using namespace scm::res;


resource_base::resource_base()
{
}

resource_base::resource_base(const resource_base& res)
  : _resource(res._resource),
    _manager(res._manager)
{
}

resource_base::resource_base(const resource_base::resource_ptr&  res,
                             const resource_base::manager_ptr&   mgr)
  : _resource(res),
    _manager(mgr)
{
}

resource_base::~resource_base()
{
    release_instance();
}

const resource_base& resource_base::operator=(const resource_base& rhs)
{
    release_instance();

    _manager    = rhs._manager;
    _resource   = rhs._resource;

    register_instance();

    return (*this);
}

resource_base::operator bool() const
{
    return (_resource.lock());
}

bool resource_base::operator !() const
{
    return (!_resource.lock());
}

void resource_base::swap(resource_base& ref)
{
    std::swap(_manager,  ref._manager);
    std::swap(_resource, ref._resource);
}

void resource_base::register_instance()
{
    if (boost::shared_ptr<resource_manager_base> m = _manager.lock()) {
        m->register_instance(*this);
    }
}

void resource_base::release_instance()
{
    if (boost::shared_ptr<resource_manager_base> m = _manager.lock()) {
        m->release_instance(*this);
    }
}

} // namespace res
} // namespace scm
