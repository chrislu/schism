
#ifndef SCM_CORE_IO_LARGE_FILE_DEVICE_H_INCLUDED
#define SCM_CORE_IO_LARGE_FILE_DEVICE_H_INCLUDED

#include <string>

#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/positioning.hpp>

#include <boost/shared_ptr.hpp>      

#include <scm/core/int_types.h>
#include <scm/core/io/file.h>

namespace scm {
namespace io {

namespace detail {

const scm::uint32   default_read_write_buffer_size = 65536u;

} // namespace detail

template <typename char_t>
class large_file
{
public:
    typedef char_t  char_type;
    struct category : public boost::iostreams::seekable_device_tag,
                      public boost::iostreams::closable_tag,
                      public boost::iostreams::optimally_buffered_tag {};

    // ctor / dtor
    large_file();
    large_file(const std::string&       file_path,
               std::ios_base::openmode  open_mode = std::ios_base::in | std::ios_base::out,
               bool                     disable_system_cache = true,
               scm::uint32              read_write_buffer_size = detail::default_read_write_buffer_size);
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
                                 scm::uint32                read_write_buffer_size = detail::default_read_write_buffer_size);

    bool                    is_open() const;
    void                    close();
    std::streamsize         optimal_buffer_size() const;

private:
    boost::shared_ptr<file>     _impl;

}; // class large_file


template <typename char_t>
class large_file_source : private large_file<char_t>
{
public:
    typedef char_t  char_type;
    struct category : public boost::iostreams::input_seekable,
                      public boost::iostreams::device_tag,
                      public boost::iostreams::closable_tag,
                      public boost::iostreams::optimally_buffered_tag {};

    // required functions for stream access
    using large_file<char_t>::read;
    using large_file<char_t>::seek;
    using large_file<char_t>::is_open;
    using large_file<char_t>::close;
    using large_file<char_t>::optimal_buffer_size;

    large_file_source(const std::string&       file_path,
                      std::ios_base::openmode  open_mode = std::ios_base::in,
                      bool                     disable_system_cache = true,
                      scm::uint32              read_write_buffer_size = detail::default_read_write_buffer_size)
        : large_file<char_t>(file_path,
                             open_mode & ~std::ios_base::out,
                             disable_system_cache,
                             read_write_buffer_size)
    {
    }
    large_file_source(const large_file_source& rhs)
        : large_file<char_t>(rhs)
    {
    }
    void                    open(const std::string&         file_path,
                                 std::ios_base::openmode    open_mode = std::ios_base::in,
                                 bool                       disable_system_cache = true,
                                 scm::uint32                read_write_buffer_size = detail::default_read_write_buffer_size)
    {
        large_file<char_t>::open(file_path,
                                 open_mode & ~std::ios_base::out,
                                 disable_system_cache,
                                 read_write_buffer_size);
    }

}; // class large_file_source

template <typename char_t>
class large_file_sink : private large_file<char_t>
{
public:
    typedef char_t  char_type;
    struct category : public boost::iostreams::output_seekable,
                      public boost::iostreams::device_tag,
                      public boost::iostreams::closable_tag,
                      public boost::iostreams::optimally_buffered_tag {};

    // required functions for stream access
    using large_file<char_t>::write;
    using large_file<char_t>::seek;
    using large_file<char_t>::is_open;
    using large_file<char_t>::close;
    using large_file<char_t>::optimal_buffer_size;

    large_file_sink(const std::string&       file_path,
                      std::ios_base::openmode  open_mode = std::ios_base::out,
                      bool                     disable_system_cache = true,
                      scm::uint32              read_write_buffer_size = detail::default_read_write_buffer_size)
        : large_file<char_t>(file_path,
                             open_mode & ~std::ios_base::in,
                             disable_system_cache,
                             read_write_buffer_size)
    {
    }
    large_file_sink(const large_file_sink& rhs)
        : large_file<char_t>(rhs)
    {
    }
    void                    open(const std::string&         file_path,
                                 std::ios_base::openmode    open_mode = std::ios_base::out,
                                 bool                       disable_system_cache = true,
                                 scm::uint32                read_write_buffer_size = detail::default_read_write_buffer_size)
    {
        large_file<char_t>::open(file_path,
                                 open_mode & ~std::ios_base::in,
                                 disable_system_cache,
                                 read_write_buffer_size);
    }

}; // class large_file_sink

} // namespace io
} // namespace scm

#include "large_file_device.inl"

#endif // SCM_CORE_IO_LARGE_FILE_DEVICE_H_INCLUDED
