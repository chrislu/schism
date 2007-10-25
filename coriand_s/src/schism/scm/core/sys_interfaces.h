
#ifndef BASIC_SYSTEM_INTERFACES_H_INCLUDED
#define BASIC_SYSTEM_INTERFACES_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace core {

class __scm_export(core) system : public boost::noncopyable
{
public:
    system() : _initialized(false) {}
    virtual ~system() {}

    virtual bool            initialize()    = 0;
    virtual bool            shutdown()      = 0;

    bool                    is_initialized() const { return (_initialized); }

protected:
    bool                    _initialized;

}; // class system

class __scm_export(core) system_refreshable : public scm::core::system
{
public:
    system_refreshable() : system() {}
    virtual ~system_refreshable() {}

    using scm::core::system::initialize;
    using scm::core::system::shutdown;

    virtual bool            frame()         = 0;

}; // class system_refreshable

} // namespace system
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // BASIC_SYSTEM_INTERFACES_H_INCLUDED

