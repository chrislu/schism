
namespace scm {
namespace con {

// define pass through std::ostream operator << to _output_buffer
template<class T> console_out_stream& operator << (console_out_stream& con, const T& rhs) {
    con._stream << rhs;
    con.emit_stream_updated_signal();
    return (con);
}

inline console_out_stream& operator << (console_out_stream& con, std::ostream& (*_Pfn)(std::ostream&)) {
    con._stream << _Pfn;
    con.emit_stream_updated_signal();
    return (con);
}

inline console_out_stream& operator << (console_out_stream& con, const log_level& level) {
    con._log_level = level.get_level();
    return (con);
}

} // namespace con
} // namespace scm
