
#ifndef SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED
#define SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED

namespace scm {
namespace time {

template<class timer_t>
class accumulate_timer
{
public:
    typedef timer_t                                 timer_type;
    typedef typename timer_type::duration_type      duration_type;

public:
    accumulate_timer();

    void                    start();
    void                    stop();
    void                    reset();

    const duration_type&    accumulated_duration() const;
    unsigned                accumulation_count() const;

    duration_type           average_duration() const;

protected:
    duration_type           _accumulated_duration;
    unsigned                _accumulation_count;

    timer_type              _timer;
};

} // namespace time
} // namespace scm

#include "accumulate_timer.inl"

#endif // SCM_TIME_ACCUMULATE_TIMER_H_INCLUDED
