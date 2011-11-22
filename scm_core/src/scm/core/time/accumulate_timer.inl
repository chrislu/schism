
namespace scm {
namespace time {

template<class timer_t>
accumulate_timer<timer_t>::accumulate_timer()
  : _accumulated_duration(duration_type()),
    _accumulation_count(0u)
{
    _timer.start();
    _timer.stop();
}

template<class timer_t>
void
accumulate_timer<timer_t>::start()
{
    _timer.start();
}

template<class timer_t>
void
accumulate_timer<timer_t>::stop()
{
    _timer.stop();

    _accumulated_duration += _timer.get_time();
    ++_accumulation_count;
}

template<class timer_t>
void
accumulate_timer<timer_t>::reset()
{
    _accumulated_duration   = duration_type();
    _accumulation_count     = 0u;
}

template<class timer_t>
const typename accumulate_timer<timer_t>::duration_type&
accumulate_timer<timer_t>::accumulated_duration() const
{
    return (_accumulated_duration);
}

template<class timer_t>
unsigned
accumulate_timer<timer_t>::accumulation_count() const
{
    return (_accumulation_count);
}

template<class timer_t>
typename accumulate_timer<timer_t>::duration_type
accumulate_timer<timer_t>::average_duration() const
{
    if (_accumulation_count > 0) {
        return (_accumulated_duration / _accumulation_count);
    }
    else {
        return (duration_type());
    }
}

template<class timer_t>
typename accumulate_timer<timer_t>::duration_type
accumulate_timer<timer_t>::last_time() const
{
    return _timer.get_time();
}

} // namespace time
} // namespace scm
