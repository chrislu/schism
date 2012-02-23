
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "file_core_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <vector>
#include <cassert>

#include <scm/core/platform/windows.h>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/log.h>
#include <scm/core/math/math.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>
#include <scm/core/utilities/foreach.h>

// TODO
// have one reader thread runnable/running which only sends out the read requests when one is put into the queue
// this way as soon as a request is free it can be resend if necessary, the calling thread only starts the read thread
// and waits for requests to be filled, after handling a filled request it is put back into the queue for open read
// requests and a condition variable for the read thread is triggered

namespace scm {
namespace io {

namespace detail {

struct overlapped_ext : public OVERLAPPED
{
    overlapped_ext(const file_core::size_type size);
    ~overlapped_ext(); // needs to be non-virtual

    void                                            position(const file_core::offset_type pos);
    file_core::offset_type                          position() const;

    void                                            bytes_to_process(const file_core::size_type size);
    file_core::size_type                            bytes_to_process() const;

    const scm::shared_ptr<file_core::char_type>&    buffer() const;

private:
    file_core::size_type                            _bytes_to_process;
    scm::shared_ptr<file_core::char_type>           _rw_buffer;
}; // struct overlapped_ext

typedef std::queue<request_ptr>                             request_ptr_queue;
typedef std::map<request_ptr::element_type*, request_ptr>   request_ptr_map;

overlapped_ext::overlapped_ext(const file_core::size_type size)
  : _bytes_to_process(0)
{
    hEvent = 0;
    _rw_buffer.reset(static_cast<file_core::char_type*>(VirtualAlloc(0, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)),
                                                        boost::bind<BOOL>(VirtualFree, _1, 0, MEM_RELEASE));
    assert(_rw_buffer);
}

overlapped_ext::~overlapped_ext()
{
    _rw_buffer.reset();
}

void
overlapped_ext::position(const file_core::offset_type pos)
{
    LARGE_INTEGER   l;
    l.QuadPart = pos;

    this->Offset        = l.LowPart;
    this->OffsetHigh    = l.HighPart;
}

file_core::offset_type
overlapped_ext::position() const
{
    LARGE_INTEGER   l;

    l.LowPart           = this->Offset;
    l.HighPart          = this->OffsetHigh;

    return (l.QuadPart);
}

void
overlapped_ext::bytes_to_process(const file_core::size_type size)
{
    _bytes_to_process = size;
}

file_core::size_type
overlapped_ext::bytes_to_process() const
{
    return (_bytes_to_process);
}

const scm::shared_ptr<file_core::char_type>&
overlapped_ext::buffer() const
{
    return (_rw_buffer);
}

struct io_result
{
    io_result() : _bytes_processed(0), _key(0), _ovl(0) {}

    DWORD           _bytes_processed;
    ULONG_PTR       _key;
    overlapped_ext* _ovl;
};

} // namespace detail

file_core_win32::file_core_win32()
  : file_core()
{
}

file_core_win32::~file_core_win32()
{
}

bool
file_core_win32::open(const std::string&       file_path,
                      std::ios_base::openmode  open_mode,
                      bool                     disable_system_cache,
                      scm::uint32              read_write_buffer_size,
                      scm::uint32              read_write_asynchronous_requests)
{
    using namespace boost::filesystem;

    path            input_file_path(file_path);
    path            complete_input_file_path(system_complete(input_file_path));

    bool            input_file_exists = exists(complete_input_file_path);
    std::string     input_root_path = complete_input_file_path.root_name().string();

    // translate open mode to access mode
    DWORD desired_access = 0;
    DWORD read_write_buffer_access = 0;

    if (open_mode & std::ios_base::in) {
        desired_access              |= GENERIC_READ;
        read_write_buffer_access    =  PAGE_READWRITE;
    }
    if (open_mode & std::ios_base::out) {
        desired_access              |= GENERIC_WRITE;
        read_write_buffer_access    =  PAGE_READWRITE;
    }

    // no async writes supported
    //if (   open_mode & std::ios_base::out
    //    && disable_system_cache)
    //{
    //    scm::err() << log::error
    //               << "file_win::open(): "
    //               << "illegal open mode ("
    //               << std::hex << open_mode
    //               << ") write in compination with async read not supported"
    //               << "(on file '" << file_path << "')" << log::end;

    //    return (false);
    //}

    // share mode
    DWORD share_mode = FILE_SHARE_READ;

    // translate open mode to creation modes
    DWORD creation_disposition = 0;

    if (input_file_exists) {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = OPEN_ALWAYS;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = TRUNCATE_EXISTING;
        }
        else {
            scm::err() << log::error
                       << "file_win::open(): "
                       << "illegal open mode "
                       << std::hex << open_mode
                       << " on file '" << file_path << "'" << log::end;

            return (false);
        }
    }
    else {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = CREATE_NEW;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = CREATE_NEW;
        }
        else {
            scm::err() << log::error
                       << "file_win::open(): "
                       << "illegal open mode "
                       << std::hex << open_mode
                       << " on file '" << complete_input_file_path.string() << "'" << log::end;

            return (false);
        }
    }

    // file attributes
    DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;

    if (disable_system_cache) {
        flags_and_attributes |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
    }

    // use shared pointer to manage the close in case we miss it somehow
    _file_handle.reset(CreateFile(complete_input_file_path.string().c_str(),
                                  desired_access,
                                  share_mode,
                                  0,
                                  creation_disposition,
                                  flags_and_attributes,
                                  0),
                       boost::bind<BOOL>(CloseHandle, _1));

    if (_file_handle.get() == INVALID_HANDLE_VALUE) {
        scm::err() << log::error
                   << "file_win::open(): "
                   << "error creating/opening file:  "
                   << "'" << complete_input_file_path.string() << "'" << log::end;

        return (false);
    }

    // retrieve the sector size information
    DWORD   sectors_per_cluster;
    DWORD   bytes_per_sector;
    DWORD   free_clusters;
    DWORD   total_clusters;
    if (GetDiskFreeSpace(input_root_path.c_str(),
                         &sectors_per_cluster,
                         &bytes_per_sector,
                         &free_clusters,
                         &total_clusters) == FALSE) {

        scm::err() << log::error
                    << "file_win::open(): "
                    << "error retrieving sector size information "
                    << "on device of file '" << complete_input_file_path.string() << "'" << log::end;

        return (false);
    }

    _volume_sector_size = bytes_per_sector;

    assert(_volume_sector_size != 0);

    if (disable_system_cache) {

        // create completion io port for asynchronous read/write operations
        _completion_port.reset(CreateIoCompletionPort(_file_handle.get(), NULL, 0 /*generate key*/, 0),
                               boost::bind<BOOL>(CloseHandle, _1));

        if (_completion_port.get() == NULL) {
            scm::err() << log::error
                       << "file_win::open(): "
                       << "error creating completion io port:  "
                       << "'" << complete_input_file_path.string() << "'" << log::end;

            return (false);
        }

        // calculate the correct read write buffer size (round up to full multiple of bytes per sector)
        _async_request_buffer_size  = static_cast<scm::int32>(vss_align_ceil(read_write_buffer_size));
        _async_requests             = read_write_asynchronous_requests;

        assert(_async_request_buffer_size % _volume_sector_size == 0);
        assert(read_write_buffer_access != 0);
    }

    if (   open_mode & std::ios_base::ate
        || open_mode & std::ios_base::app) {

        _position = actual_file_size();
    }

    _file_path  = complete_input_file_path.string();
    _open_mode  = open_mode;
    _file_size  = actual_file_size();

    return (true);
}

bool
file_core_win32::is_open() const
{
    if (_file_handle) {
        return ((_file_handle.get() == INVALID_HANDLE_VALUE) ? false : true);
    }
    else {
        return (false);
    }
}

void
file_core_win32::close()
{
    if (is_open()) {
        // if we are non system buffered, it is possible to be too large
        // because of volume sector size alignment restrictions
        if (   async_io_mode()
            && _open_mode & std::ios_base::out) {
            if (_file_size != actual_file_size()) {

                _file_handle.reset();
                _file_handle.reset(CreateFile(_file_path.c_str(),
                                              GENERIC_WRITE,
                                              PAGE_READONLY,
                                              0,
                                              OPEN_EXISTING,
                                              FILE_ATTRIBUTE_NORMAL,
                                              0),
                                   boost::bind<BOOL>(CloseHandle, _1));

                if (_file_handle.get() == INVALID_HANDLE_VALUE) {
                    scm::err() << log::error
                               << "file_win::close(): "
                               << "error opening file for truncating end: "
                               << "'" << _file_path << "'" << log::end;
                    throw std::ios_base::failure(  std::string("file_win::close(): error opening file for truncating end: ")
                                                 + _file_path);
                }
                set_file_pointer(_file_size);
                if (SetEndOfFile(_file_handle.get()) == 0) {
                    scm::err() << log::error
                               << "file_win::close(): "
                               << "error truncating end of file: "
                               << "'" << _file_path << "'" << log::end;
                    throw std::ios_base::failure(  std::string("file_win::close(): error truncating end of file: ")
                                                 + _file_path);
                }
            }
        }
    }

    // reset read write buffer invoking VirtualFree
    reset_values();
}

file_core_win32::size_type
file_core_win32::read(void*         output_buffer,
                      offset_type   start_position,
                      size_type     num_bytes_to_read)
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);
    assert(_position >= 0);

    using namespace scm;

    uint8*      output_byte_buffer  = reinterpret_cast<uint8*>(output_buffer);
    int64       bytes_read          = 0;

    // non system buffered read operation
    //if (_rw_buffer_size > 0) {
    if (async_io_mode()) {
        bytes_read = read_async(output_buffer, start_position, num_bytes_to_read);
    }
    // normal system buffered operation
    else {
        _position = start_position;
        if (!set_file_pointer(_position)) {
            scm::err() << log::error
                       << "file_win::read(): "
                       << "unable to set file pointer to current position " << log::end;
            return (0);
            //throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unable to set file pointer to current position");
        }

        DWORD   file_bytes_read       = 0;

        if (ReadFile(_file_handle.get(), output_byte_buffer, static_cast<DWORD>(num_bytes_to_read), &file_bytes_read, 0) == 0) {
            scm::err() << log::error
                       << "file_win::read(): "
                       << "error reading from file " << _file_path << log::end;
            return (0);
            //throw std::ios_base::failure("large_file_device_windows<char_type>::read(): error reading from file");
        }

        if (file_bytes_read == 0) {
            // eof
            return (-1);
        }
        if (file_bytes_read <= num_bytes_to_read) {
            _position           += file_bytes_read;
            bytes_read           = file_bytes_read;
        }
        else {
            //throw std::ios_base::failure("large_file_device_windows<char_type>::read(): unknown error reading from file");
            scm::err() << log::error
                       << "file_win::read(): "
                       << "unknown error reading from file " << _file_path << log::end;
            return (0);
        }
    }

    assert(bytes_read > 0);

    return (bytes_read);
}

file_core_win32::size_type
file_core_win32::write(const void* input_buffer,
                       offset_type start_position,
                       size_type   num_bytes_to_write)
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);
    assert(_position >= 0);

    using namespace scm;

    if (_open_mode & std::ios_base::app) {
        _position = _file_size;
    }

    const uint8*    input_byte_buffer   = reinterpret_cast<const uint8*>(input_buffer);
    int64           bytes_written       = 0;

    // non system buffered read operation
    if (async_io_mode()) {
        //scm::err() << log::error
        //           << "file_win::write(): "
        //           << "file was opened for async read operations (async write not supported)" << log::end;
        //return (0);
        bytes_written = write_async(input_buffer, start_position, num_bytes_to_write);
    }
    // normal system buffered operation
    else {
        _position = start_position;
        if (!set_file_pointer(_position)) {
            return (0);
        }

        DWORD   file_bytes_written  = 0;

        if (WriteFile(_file_handle.get(),
                      input_byte_buffer,
                      static_cast<DWORD>(num_bytes_to_write),
                      &file_bytes_written,
                      0) == 0) {
            scm::err() << log::error
                       << "file_win::write(): "
                       << "error writing to file " << _file_path << log::end;
            return (0);
        }

        if (file_bytes_written <= num_bytes_to_write) {
            _position           += file_bytes_written;
            bytes_written        = file_bytes_written;
        }
        else {
            scm::err() << log::error
                       << "file_win::write(): "
                       << "unknown error writing to file " << _file_path << log::end;
        }
    }

    return (bytes_written);
}

bool
file_core_win32::flush_buffers() const
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);

    return (FlushFileBuffers(_file_handle.get()) == TRUE ? true : false);
}

file_core_win32::offset_type
file_core_win32::set_end_of_file()
{
    if (is_open()) {
        if (   async_io_mode()
            && _open_mode & std::ios_base::out) {
            // we use non system buffered access
            // we need to reopen the file for the truncation
            _file_handle.reset();
            _file_handle.reset(CreateFile(_file_path.c_str(),
                                          GENERIC_WRITE,
                                          PAGE_READONLY,
                                          0,
                                          OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL,
                                          0),
                               boost::bind<BOOL>(CloseHandle, _1));

            if (_file_handle.get() == INVALID_HANDLE_VALUE) {
                scm::err() << log::error
                           << "file_win::set_end_of_file(): "
                           << "error opening file for truncating end: "
                           << _file_path << log::end;

                close();
                return (-1);
            }

            // set file pointer to truncate location
            set_file_pointer(_position);
            if (SetEndOfFile(_file_handle.get()) == 0) {
                scm::err() << log::error
                           << "file_win::set_end_of_file(): "
                           << "error truncating end of file: "
                           << "position " << std::hex << _position << " file "
                           << _file_path << log::end;

                close();
                return (-1);
            }

            _file_size = _position;

            // reopen file with non system buffered access

            std::ios_base::open_mode    reopen_mode = _open_mode;

            reopen_mode &= ~std::ios_base::trunc;

            _file_handle.reset();
            if (!open(_file_path, reopen_mode, true, _async_request_buffer_size, _async_requests)) {
                scm::err() << log::error
                           << "file_win::set_end_of_file(): "
                           << "error reopening file: "
                           << _file_path << log::end;
                close();
                return (-1);
            }
        }
        else {
            set_file_pointer(_position);
            if (SetEndOfFile(_file_handle.get()) == 0) {
                scm::err() << log::error
                           << "file_win::set_end_of_file(): "
                           << "error truncating end of file: "
                           << "position " << std::hex << _position << " file "
                           << _file_path << log::end;

                close();
                return (-1);
            }
        }

        return (_position);
    }

    return (-1);
}


file_core_win32::size_type
file_core_win32::read_async(void*       output_buffer,
                            offset_type start_position,
                            size_type   num_bytes_to_read)
{
    assert(async_io_mode());

    using detail::overlapped_ext;
    using detail::request_ptr;
    using detail::request_ptr_queue;
    using detail::request_ptr_map;

    request_ptr_queue       free_requests;
    request_ptr_map         running_requests;

    char* output_byte_buffer   = reinterpret_cast<char*>(output_buffer);

    _position   = start_position;

    size_type   position_vss            = vss_align_floor(_position);
    size_type   file_size_vss           = vss_align_ceil(_file_size);
    size_type   bytes_to_read_vss       = vss_align_ceil(math::min(_position  - position_vss + num_bytes_to_read,
                                                             _file_size - position_vss));

    size_type   bytes_read              = 0;
    size_type   read_end_position_vss   = position_vss + bytes_to_read_vss;
    size_type   next_read_request_pos   = position_vss;

    scm::int32  allocate_requests       = scm::math::min<scm::int32>(_async_requests, static_cast<scm::int32>(bytes_to_read_vss / _async_request_buffer_size + 1));

    //scm::out() << "allocate_requests: " << allocate_requests << log::end;

    // allocate the request structs
    for (scm::int32 i = 0; i < allocate_requests; ++i) {
        request_ptr new_request(new overlapped_ext(_async_request_buffer_size));
        free_requests.push(new_request);
    }

    scm::time::accum_timer<scm::time::high_res_timer>  request_processing_timer;

    do {
        // fill up request queue
        while (!free_requests.empty() && next_read_request_pos < read_end_position_vss) {
            // retrieve a free request structure
            request_ptr  read_request_ovl = free_requests.front();
            free_requests.pop();

            size_type bytes_left            = read_end_position_vss - next_read_request_pos;
            size_type request_bytes_to_read = scm::math::min<size_type>(bytes_left, _async_request_buffer_size);
            DWORD     request_bytes_read    = 0;

            // setup overlapped structure
            read_request_ovl->position(next_read_request_pos);
            read_request_ovl->bytes_to_process(request_bytes_to_read);

            next_read_request_pos += request_bytes_to_read;
            running_requests.insert(request_ptr_map::value_type(read_request_ovl.get(), read_request_ovl));

            if (!read_async_request(read_request_ovl)) {
                cancel_async_io();
                return (bytes_read);
            }

            assert(free_requests.size() + running_requests.size() == allocate_requests);
        }

        // ok now wait for requests to be filled
        if (!running_requests.empty()) {
            std::vector<detail::io_result>  results;

            if (!query_async_results(results, allocate_requests)) {
                cancel_async_io();
                return (bytes_read);
            }

            assert(!results.empty());

            // evaluate io results
            foreach (const detail::io_result& result, results) {

                //request_processing_timer.start();

                if (result._bytes_processed != result._ovl->bytes_to_process()) {
                    if (result._ovl->position() + result._bytes_processed < _file_size) {
                        scm::err() << log::error
                                   << "file_core_win32::read_async(): read result with different than requested length "
                                   << "(requested: " << result._ovl->bytes_to_process()
                                   << ", read: " << result._bytes_processed << ")" << log::end;

                        cancel_async_io();
                        return (bytes_read);
                    }
                }
                // copy the data from the ovl buffer to the outbuffer
                size_type   target_off      = result._ovl->position() - _position;
                size_type   copy_write_off  = math::max<size_type>(0,  target_off);
                size_type   copy_read_off   = math::max<size_type>(0, -target_off);
                size_type   copy_read_bytes = math::min<size_type>(result._bytes_processed - copy_read_off, num_bytes_to_read - copy_write_off);

                char_type*       copy_dst   = output_byte_buffer          + copy_write_off;
                const char_type* copy_src   = result._ovl->buffer().get() + copy_read_off;

                if (memcpy(copy_dst, copy_src, copy_read_bytes) != copy_dst) {
                    scm::err() << log::error
                               << "file_core_win32::read_async(): error copying to destination buffer" << log::end;

                    cancel_async_io();
                    return (bytes_read);
                }

                bytes_read += copy_read_bytes;

                // find our ovl structure in the map
                // add the pointer to the free list and remove it from the used map
                request_ptr_map::iterator   result_request = running_requests.find(result._ovl);

                if (result_request != running_requests.end()) {
                    free_requests.push(result_request->second);
                    running_requests.erase(result_request);
                }
                else {
                    scm::err() << log::error
                               << "file_core_win32::read_async(): error finding result read request in running request list" << log::end;

                    cancel_async_io();
                    return (bytes_read);
                }
                assert(free_requests.size() + running_requests.size() == allocate_requests);

                //request_processing_timer.stop();
            }
        }
    } while(   bytes_read < num_bytes_to_read
            && !(running_requests.empty() && next_read_request_pos >= read_end_position_vss));

    if (bytes_read == num_bytes_to_read) {
        _position = _position + bytes_read;
    }

    //double avg_request_processing_time = scm::time::to_milliseconds(request_processing_timer.accumulated_duration())
    //                                     / request_processing_timer.accumulation_count();

    //scm::out() << std::fixed << "avg request processing time: " << avg_request_processing_time << "msec" << log::end;

    return (bytes_read);
}

bool
file_core_win32::read_async_request(const detail::request_ptr& req) const
{
    DWORD request_bytes_read = 0;

    if (ReadFile(_file_handle.get(),
                 req->buffer().get(),
                 static_cast<DWORD>(req->bytes_to_process()),
                 &request_bytes_read,
                 req.get()) == 0)
    {
        if (GetLastError() != ERROR_IO_PENDING) {
            scm::err() << log::error
                       << "file_core_win32::read_async_request(): "
                       << "error starting read request "
                       << "(file: "      << _file_path
                       << ", position: " << std::hex << "0x" << req->position()
                       << ", length: "   << std::dec << req->bytes_to_process() << ")" << log::end;
            return (false);
        }
    }

    return (true);
}

file_core_win32::size_type
file_core_win32::write_async(const void* input_buffer,
                             offset_type start_position,
                             size_type   num_bytes_to_write)
{
    assert(async_io_mode());

    using detail::overlapped_ext;
    using detail::request_ptr;
    using detail::request_ptr_queue;
    using detail::request_ptr_map;

    request_ptr_queue       free_requests;
    request_ptr_map         running_requests;

    const char* input_byte_buffer  = reinterpret_cast<const char*>(input_buffer);

    _position = start_position;

    size_type   position_vss            = vss_align_floor(_position);
    size_type   position_end_vss        = vss_align_ceil(_position + num_bytes_to_write);

    size_type   file_size_vss           = vss_align_ceil(_file_size);
    size_type   bytes_to_write_vss      = position_end_vss - position_vss;

    size_type   begin_sec               = position_vss;
    size_type   begin_sec_prefetch      = position_vss - _position;
    size_type   end_sec                 = vss_align_floor(_position + num_bytes_to_write);
    size_type   end_sec_prefetch        = position_end_vss - end_sec;

    size_type   bytes_written           = 0;
    size_type   next_write_request_pos  = position_vss;

    if (begin_sec_prefetch != 0 &&
        end_sec_prefetch   != 0) {
        scm::err() << log::error
                   << "file_core_win32::write_async(): trying to write shit that is not vss aligned!"
                   << " file: " << _file_path << ")" << log::end;
        return (0);
    }

    scm::int32 allocate_requests = scm::math::min<scm::int32>(_async_requests, static_cast<scm::int32>(bytes_to_write_vss / _async_request_buffer_size + 1));

    //scm::out() << "allocate_requests: " << allocate_requests << log::end;

    // allocate the request structs
    for (scm::int32 i = 0; i < allocate_requests; ++i) {
        request_ptr new_request(new overlapped_ext(_async_request_buffer_size));
        free_requests.push(new_request);
    }

    do {
        // fill up request queue
        while (!free_requests.empty() && next_write_request_pos < position_end_vss) {
            // retrieve a free request structure
            request_ptr  write_request_ovl = free_requests.front();
            free_requests.pop();

            size_type bytes_left                = position_end_vss - next_write_request_pos;
            size_type request_bytes_to_write    = scm::math::min<size_type>(bytes_left, _async_request_buffer_size);
            DWORD     request_bytes_written     = 0;

            // setup overlapped structure
            write_request_ovl->position(next_write_request_pos);
            write_request_ovl->bytes_to_process(request_bytes_to_write);

            // copy the request data to the request buffer
            size_type   copy_read_off       = write_request_ovl->position() - position_vss;
            size_type   copy_write_off      = 0;
            size_type   copy_write_bytes    = request_bytes_to_write;

            const char_type* copy_src       = input_byte_buffer                 + copy_read_off;
            char_type*       copy_dst       = write_request_ovl->buffer().get() + copy_write_off;

            if (memcpy(copy_dst, copy_src, copy_write_bytes) != copy_dst) {
                scm::err() << log::error
                           << "file_core_win32::write_async(): error copying to destination buffer" << log::end;

                cancel_async_io();
                return (bytes_written);
            }

            next_write_request_pos += request_bytes_to_write;
            running_requests.insert(request_ptr_map::value_type(write_request_ovl.get(), write_request_ovl));

            if (!write_async_request(write_request_ovl)) {
                cancel_async_io();
                return (bytes_written);
            }

            assert(free_requests.size() + running_requests.size() == allocate_requests);
        }

        // ok now wait for requests to be filled
        if (!running_requests.empty()) {
            std::vector<detail::io_result>  results;

            if (!query_async_results(results, allocate_requests)) {
                cancel_async_io();
                return (bytes_written);
            }

            assert(!results.empty());

            // evaluate io results
            foreach (const detail::io_result& result, results) {
                if (result._bytes_processed != result._ovl->bytes_to_process()) {
                    scm::err() << log::error
                               << "file_core_win32::write_async(): write result with different than requested length "
                               << "(requested: " << result._ovl->bytes_to_process()
                               << ", read: " << result._bytes_processed << ")" << log::end;

                    cancel_async_io();
                    return (bytes_written);
                }

                bytes_written += result._bytes_processed;

                // find our ovl structure in the map
                // add the pointer to the free list and remove it from the used map
                request_ptr_map::iterator   result_request = running_requests.find(result._ovl);

                if (result_request != running_requests.end()) {
                    free_requests.push(result_request->second);
                    running_requests.erase(result_request);
                }
                else {
                    scm::err() << log::error
                               << "file_core_win32::write_async(): error finding result write request in running request list" << log::end;

                    cancel_async_io();
                    return (bytes_written);
                }
                assert(free_requests.size() + running_requests.size() == allocate_requests);
            }
        }
    } while(bytes_written < num_bytes_to_write);

    if (bytes_written == num_bytes_to_write) {
        _position = _position + bytes_written;
    }
    if (_file_size < _position) {
        _file_size = _position;
    }

    return (bytes_written);
}

bool
file_core_win32::write_async_request(const detail::request_ptr& req) const
{
    DWORD request_bytes_write = 0;

    if (WriteFile(_file_handle.get(),
                  req->buffer().get(),
                  static_cast<DWORD>(req->bytes_to_process()),
                  &request_bytes_write,
                  req.get()) == 0)
    {
        if (GetLastError() != ERROR_IO_PENDING) {
            scm::err() << log::error
                       << "file_core_win32::write_async_request(): "
                       << "error starting write request "
                       << "(file: "      << _file_path
                       << ", position: " << std::hex << "0x" << req->position()
                       << ", length: "   << std::dec << req->bytes_to_process() << ")" << log::end;
            return (false);
        }
    }

    return (true);
}

bool
file_core_win32::query_async_results(std::vector<detail::io_result>& results_vec,
                                     int query_max_results) const
{
    using detail::overlapped_ext;
    using detail::io_result;

#if SCM_WIN_VER >= SCM_WIN_VER_VISTA
    scm::scoped_array<OVERLAPPED_ENTRY> result_entries(new OVERLAPPED_ENTRY[query_max_results]);
    ULONG                               result_entries_fetched = 0;

    if (GetQueuedCompletionStatusEx(_completion_port.get(),
                                    result_entries.get(),
                                    query_max_results,
                                    &result_entries_fetched,
                                    INFINITE,
                                    FALSE) != 0)
    {
        //scm::out() << result_entries_fetched << log::end;
        assert(results_vec.empty());

        for (ULONG i = 0; i < result_entries_fetched; ++i) {
            io_result new_result;

            new_result._bytes_processed = result_entries[i].dwNumberOfBytesTransferred;
            new_result._key             = result_entries[i].lpCompletionKey;
            new_result._ovl             = static_cast<detail::overlapped_ext*>(result_entries[i].lpOverlapped);

            results_vec.push_back(new_result);
        }
    }
    else {
        scm::err() << log::error
                   << "file_core_win32::read_async(): GetQueuedCompletionStatusEx returned with error" << log::end;

        return (false);
    }
#else // SCM_WIN_VER >= SCM_WIN_VER_VISTA
    DWORD           result_bytes_read = 0;
    ULONG_PTR       result_key        = 0;
    overlapped_ext* result_ovl;

    if (GetQueuedCompletionStatus(_completion_port.get(),
                                  &result_bytes_read,
                                  &result_key,
                                  reinterpret_cast<LPOVERLAPPED*>(&result_ovl),
                                  INFINITE) != 0)
    {
        io_result new_result;

        new_result._bytes_processed = result_bytes_read;
        new_result._key             = result_key;
        new_result._ovl             = result_ovl;

        assert(results_vec.empty());

        results_vec.push_back(new_result);
    }
    else {
        scm::err() << log::error
                   << "file_core_win32::read_async(): GetQueuedCompletionStatus returned with error" << log::end;

        return (false);
    }
#endif // SCM_WIN_VER >= SCM_WIN_VER_VISTA

    return (true);
}


void
file_core_win32::cancel_async_io() const
{
#if SCM_WIN_VER >= SCM_WIN_VER_VISTA
    // CancelIoEx only available on Windows Vista and up
    if (CancelIoEx(_file_handle.get(), NULL) == 0) {
        scm::err() << log::error
                   << "file_core_win32::cancel_async_io(): "
                   << "error cancelling outstanding io requests "
                   << "(file: " << _file_path << ")" << log::end;
    }
#else // SCM_WIN_VER >= SCM_WIN_VER_VISTA
    if (CancelIo(_file_handle.get()) == 0) {
        scm::err() << log::error
                   << "file_core_win32::cancel_async_io(): "
                   << "error cancelling outstanding io requests "
                   << "(file: " << _file_path << ")" << log::end;
    }
#endif // SCM_WIN_VER >= SCM_WIN_VER_VISTA
}

file_core_win32::size_type
file_core_win32::actual_file_size() const
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);

    LARGE_INTEGER   cur_size_li;

    if (GetFileSizeEx(_file_handle.get(), &cur_size_li) == 0) {
        scm::err() << log::error
                   << "file_win::actual_file_size(): "
                   << "error retrieving current file size: " << _file_path << log::end;
        throw std::ios_base::failure("file_win::actual_file_size(): error retrieving current file size");
    }

    return (static_cast<size_type>(cur_size_li.QuadPart));
}

bool
file_core_win32::set_file_pointer(offset_type new_pos)
{
    assert(_file_handle);
    assert(_file_handle.get() != INVALID_HANDLE_VALUE);

    LARGE_INTEGER   position_li;

    position_li.QuadPart = new_pos;

    if (   SetFilePointer(_file_handle.get(), position_li.LowPart, &position_li.HighPart, FILE_BEGIN) == INVALID_SET_FILE_POINTER
        && GetLastError() != NO_ERROR) {
        scm::err() << log::error
                   << "file_win::set_file_pointer(): "
                   << "error setting file pointer to position "
                   << std::hex << new_pos
                   << " on file '" << _file_path << "'" << log::end;

        return (false);
    }

    return (true);
}

void
file_core_win32::reset_values()
{
    file_core::reset_values();

    _completion_port.reset();
    _file_handle.reset();
}

} // namepspace io
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
