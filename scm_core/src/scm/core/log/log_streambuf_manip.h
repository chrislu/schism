
#ifndef SCM_CORE_LOG_LOG_STREAMBUF_MANIP_H_INCLUDED
#define SCM_CORE_LOG_LOG_STREAMBUF_MANIP_H_INCLUDED

#include <cassert>
#include <ostream>
#include <stdexcept>

#include <scm/core/log/log_streambuf.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace logging {
namespace detail {

__scm_export(core) int log_streambuf_index();

} // namespace detail


class indent
{
public:
    indent(int i) : _indention(i > 0 ? i : 0) {}

    template <typename char_type, typename traits>
    std::basic_ostream<char_type, traits> &operator()(std::basic_ostream<char_type, traits>& os) const
    {
        typedef basic_log_streambuf<char_type, traits> log_streambuf_type;

        log_streambuf_type* lsb = dynamic_cast<log_streambuf_type*>(os.rdbuf());
        if (!lsb) {
            lsb = install_log_streambuf(os, detail::log_streambuf_index());
            os.register_callback(callback, detail::log_streambuf_index());
        }
        lsb->indention(_indention);

        return (os);
    }
private:
    template <typename char_type, typename traits>
    static
    basic_log_streambuf<char_type, traits>*
    install_log_streambuf(std::basic_ostream<char_type, traits>& os, int index)
    {
        typedef basic_log_streambuf<char_type, traits> log_streambuf_type;

        log_streambuf_type* log_rdbuf = new log_streambuf_type(os.rdbuf());
        os.rdbuf(log_rdbuf);
        os.pword(index) = log_rdbuf;

        assert(log_rdbuf == os.rdbuf());
        assert(log_rdbuf == os.pword(index));
        //assert(os.pword(index) == os.rdbuf());

        return (log_rdbuf);
    }

    template <typename char_type, typename traits>
    static
    void
    uninstall_log_streambuf(std::basic_ostream<char_type, traits>& os, int index)
    {
        typedef basic_log_streambuf<char_type, traits> log_streambuf_type;

        log_streambuf_type* old_ptr = static_cast<log_streambuf_type*>(os.pword(index));
        log_streambuf_type* old_ptr_rd = dynamic_cast<log_streambuf_type*>(os.rdbuf());

        assert(0 != old_ptr);
        assert(old_ptr == old_ptr_rd);

        assert(0 != os.pword(index));
        //assert(os.pword(index) == os.rdbuf());

        log_streambuf_type* log_rdbuf = static_cast<log_streambuf_type*>(os.pword(index));
        os.rdbuf(log_rdbuf->original_rdbuf());

        delete log_rdbuf;
        os.pword(index) = 0;
    }

    static void callback(std::ios_base::event ev, std::ios_base& ios_obj, int index) {
        if (ev == std::ios_base::erase_event) {
            // ok if we are here this means, we are a log_streambuf about to be deleted
            // this could mean, we get simply deleted or we are about to get copied
            assert(0 != ios_obj.pword(index));

            if (std::ostream& os = dynamic_cast<std::ostream&>(ios_obj)) {
                uninstall_log_streambuf(os, index);
            }
            else if (std::wostream& os = dynamic_cast<std::wostream&>(ios_obj)) {
                uninstall_log_streambuf(os, index);
            }
            else {
                throw std::runtime_error("scm::log::indent::callback() runtime error, failed to dynamic_cast ios_obj.");
            }
        }
        else if(ev == std::ios_base::copyfmt_event) {
            // ok here we should have gone though the erase_event and now our pword
            // has the new log_streambuf pointer
            assert(0 != ios_obj.pword(index));

            if (std::ostream& os = dynamic_cast<std::ostream&>(ios_obj)) {
                log_streambuf* copied_log_buf = static_cast<log_streambuf*>(os.pword(index));
                log_streambuf* new_log_rdbuf  = install_log_streambuf(os, index);

                new_log_rdbuf->copy_indention_attributes(*copied_log_buf);
            }
            else if (std::wostream& os = dynamic_cast<std::wostream&>(ios_obj)) {
                wlog_streambuf* copied_log_buf = static_cast<wlog_streambuf*>(os.pword(index));
                wlog_streambuf* new_log_rdbuf  = install_log_streambuf(os, index);

                new_log_rdbuf->copy_indention_attributes(*copied_log_buf);
            }
            else {
                throw std::runtime_error("scm::log::indent::callback() runtime error, failed to dynamic_cast ios_obj.");
            }
        }
    }

private:
    int _indention;
};

template <typename char_type, typename traits>
inline std::basic_ostream<char_type, traits>&
operator<<(std::basic_ostream<char_type,traits>& os, const scm::logging::indent& indention)
{
    return (indention(os));
}

} // namespace log
} // namespace scm

#endif // SCM_CORE_LOG_LOG_STREAMBUF_MANIP_H_INCLUDED
