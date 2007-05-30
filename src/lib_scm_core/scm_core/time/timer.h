
#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED

#include <scm_core/platform/platform.h>

namespace scm {
namespace time {

class __scm_export timer
{
public:
    timer();
    virtual ~timer();

    void        start();
    void        stop();

    // returns difference time in milliseconds
    double      get_time() const { return (_diff_time); }

protected:
    double      _start_time;
    double      _diff_time;

private:

}; // class timer

} // namespace core
} // namespace scm

#endif // TIMER_H_INCLUDED
