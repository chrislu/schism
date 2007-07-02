
#ifndef SCM_INPUT_TRACKER_H_INCLUDED
#define SCM_INPUT_TRACKING_H_INCLUDED

#include <string>
#include <list>

//#include <scm/input/tracking/target.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace inp {

class target;

class __scm_export(input) tracker
{
public:
    typedef std::list<target>   target_container;

public:
    tracker(const std::string& name);
    virtual ~tracker();

    virtual bool        initialize() = 0;
    virtual void        update()     = 0;
    virtual bool        shutdown()   = 0;

    const std::string&  name() const;

protected:
    target_container    _targets;

private:
    std::string         _name;

}; // class tracker

} // namespace inp
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_INPUT_TRACKER_H_INCLUDED
