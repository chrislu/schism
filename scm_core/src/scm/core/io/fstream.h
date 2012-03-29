
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_IO_LARGE_FSTREAM_H_INCLUDED
#define SCM_IO_LARGE_FSTREAM_H_INCLUDED

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/stream_buffer.hpp>

#include <scm/core/io/large_file_device.h>

namespace scm {
namespace io {

typedef boost::iostreams::stream<large_file<char> >         fstream;
typedef boost::iostreams::stream<large_file_source<char> >  ifstream;
typedef boost::iostreams::stream<large_file_sink<char> >    ofstream;

typedef boost::iostreams::stream_buffer<large_file<char> >  filebuf;

} // namespace io
} // namespace scm

#endif // SCM_IO_LARGE_FSTREAM_H_INCLUDED
