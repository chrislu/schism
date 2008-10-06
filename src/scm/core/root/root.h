
#ifndef SCM_CORE_ROOT_H_INCLUDED
#define SCM_CORE_ROOT_H_INCLUDED

#include <boost/utility.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {

class __scm_export(core) root : boost::noncopyable
{
public:
    root(int& argc, char** argv/*, app_type t = gui_application || console_application || etc.*/);
    virtual ~root();




protected:
private:
}; // class root

} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_CORE_ROOT_H_INCLUDED
