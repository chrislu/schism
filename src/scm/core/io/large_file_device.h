
#ifndef SCM_CORE_IO_LARGE_FILE_DEVICE_H_INCLUDED
#define SCM_CORE_IO_LARGE_FILE_DEVICE_H_INCLUDED

#include <string>

#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/positioning.hpp>
//#include <boost/iostreams/operations.hpp>

#include <boost/shared_ptr.hpp>      

#include <scm/core/int_types.h>

namespace scm {
namespace io {

namespace detail {

template <typename char_type> class large_file_device_impl;

} // namespace detail

template <typename char_t>
class large_file {

public:
    typedef char_t  char_type;
    struct category : public boost::iostreams::seekable_device_tag,
                      public boost::iostreams::closable_tag {};

    // ctor / dtor
    large_file(const std::string&       file_path,
               std::ios_base::openmode  open_mode = std::ios_base::in | std::ios_base::out,
               bool                     disable_system_cache = true,
               scm::uint32              read_write_buffer_size = 262144u);
    large_file(const large_file& rhs);
    virtual ~large_file();

    // required functions for stream access
    std::streamsize         read(char_type* s,          std::streamsize n);
    std::streamsize         write(const char_type* s,   std::streamsize n);

    std::streampos          seek(boost::iostreams::stream_offset    off,
                                 std::ios_base::seek_dir            way);

    void                    open(const std::string&         file_path,
                                 std::ios_base::openmode    open_mode = std::ios_base::in | std::ios_base::out,
                                 bool                       disable_system_cache = true,
                                 scm::uint32                read_write_buffer_size = 262144u);

    bool                    is_open() const;
    void                    close();

private:
    boost::shared_ptr<detail::large_file_device_impl<char_t> >      _impl;

}; // large_file


} // namespace io
} // namespace scm

#include "large_file_device.inl"

#endif // SCM_CORE_IO_LARGE_FILE_DEVICE_H_INCLUDED
