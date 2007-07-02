
#ifndef SCM_INPUT_ART_DTRACK_H_INCLUDED
#define SCM_INPUT_ART_DTRACK_H_INCLUDED

#include <cstddef>

#include <boost/scoped_ptr.hpp>

#include <scm/input/tracking/tracker.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

class DTrack;

namespace scm {
namespace inp {

class __scm_export(input) art_dtrack : public tracker
{
public:
    art_dtrack();
    virtual ~art_dtrack();

    bool                        initialize();
    void                        update();
    bool                        shutdown();

protected:

private:
    const boost::scoped_ptr<DTrack>   _dtrack;
    std::size_t                 _listening_port;
    std::size_t                 _timeout;

}; // class art_dtrack

} // namespace inp
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_INPUT_ART_DTRACK_H_INCLUDED
