
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <hash_map>
#include <vector>
#include <map>
#include <sstream>
#include <string>

#include <boost/unordered_map.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

#include <scm/core.h>
#include <scm/core/time/high_res_timer.h>

struct somesome {
    unsigned a;
    unsigned b;
    bool operator == (const somesome& rhs) const {
        return (a == rhs.a && b == rhs.b);
    }
};

std::size_t hash_value(const somesome& s) {
    scm::size_t seed = 0;

    boost::hash_combine(seed, s.a);
    boost::hash_combine(seed, s.b);

    return (seed);
}

struct hasher_some {
    std::size_t operator()(const somesome& s) const {
        scm::size_t seed = 0;

        boost::hash_combine(seed, s.a);
        boost::hash_combine(seed, s.b);

        return (seed);
    }
};

//typedef std::string key_type;
//typedef std::size_t key_type;
typedef somesome key_type;

std::vector<key_type> keys;

template<typename T>
void timemap(const int n, const int r=7, bool single_results = false) {
	std::map<std::string, std::vector<double> > timings;

    scm::time::high_res_timer timer;

	for (int j=0 ; j<r ; ++j) {
		T hsh;

		timer.start();
		// Insert every 2nd number in the sequence
		for (int i = 0; i < n; i+=2) {
			hsh[keys[i]] = i*2;
        }
        timer.stop();
        timings["1: Insert"].push_back(scm::time::to_milliseconds(timer.get_time()));

		timer.start();
		// Lookup every number in the sequence
		for (int i = 0; i < n; ++i) {
			hsh.find(keys[i]);
		}
        timer.stop();
		timings["2: Lookup"].push_back(scm::time::to_milliseconds(timer.get_time()));

		timer.start();
		// Iterate over the entries
		for (typename T::iterator it = hsh.begin(); it != hsh.end(); ++it) {
			int x = it->second;
			++x;
		}
        timer.stop();
		timings["3: Iterate"].push_back(scm::time::to_milliseconds(timer.get_time()));

		timer.start();
		// Erase the entries
		for (int i = 0; i < n; i+=2) {
			hsh.erase(keys[i]);
		}
        timer.stop();
		timings["4: Erase"].push_back(scm::time::to_milliseconds(timer.get_time()));
	}

	for (std::map<std::string,std::vector<double> >::iterator it=timings.begin() ; it!=timings.end() ; ++it) {
		double sum = 0.0;
		std::cout << it->first << "\t(";
	    for (int i = 1; i < r; ++i) {
	    	sum += it->second.at(i);
            if (single_results) {
		    	std::cout << std::fixed << std::setprecision(4) << it->second.at(i) << "msec";
                if (i != r-1) {
                    std::cout << " ";
                }
		    }
        }
		std::cout << ") " << sum/double(r-1) << "msec" << std::endl;
	}
}

int main(int argc, char **argv)
{
#ifndef NDEBUG
    std::cout << "Debug" << std::endl;
#else
    std::cout << "Release" << std::endl;
#endif
    //std::cout << "IDL: " << _ITERATOR_DEBUG_LEVEL << std::endl;
    std::cout << "SCL: " << _SECURE_SCL << std::endl;
    std::cout << "HID: " << _HAS_ITERATOR_DEBUGGING << std::endl;

    std::wcout << "SCL: " << _SECURE_SCL << std::endl;
    std::wcout << "HID: " << _HAS_ITERATOR_DEBUGGING << std::endl;

    scm::shared_ptr<scm::core>      scm_core(new scm::core(argc, argv));

    std::string alpha="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
	//string alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    boost::mt19937          generator(static_cast<unsigned int>(std::time(0)));
	boost::uniform_int<>    uni_int_dist(0, alpha.length());
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > char_gen(generator, uni_int_dist);

	boost::uniform_int<>    uni_int_dist2(5, 10);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > len_gen(generator, uni_int_dist2);

    boost::uniform_int<>    uni_int_dist3(0, (std::numeric_limits<std::size_t>::max)());
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > num_gen(generator, uni_int_dist3);

	int n = 10000;
    int i = 10000;
	keys.reserve(n);
	keys.resize(n);
	std::stringstream ss;
	for (int i=0 ; i<n ; ++i) {
#if 0
        keys[i].clear();

        int l = len_gen();
        for (int k = 0; k < l; ++k) {
            keys[i].push_back(alpha[char_gen()]);
        }
        //std::cout << strings[i] << std::endl;
        assert(strings[i].size() == l);
#elif 0
        keys[i] = num_gen();
#else
        keys[i].a = num_gen();
        keys[i].b = num_gen();
#endif
	}

	std::cout << "Timing boost::unordered_map<std::string, int>" << std::endl;
	timemap<boost::unordered_map<key_type, int> >(n, i);
	std::cout << std::endl;

	std::cout << "Timing std::tr1::unordered_map<std::string, int>" << std::endl;
	timemap<std::unordered_map<key_type, int, hasher_some> >(n, i);
	std::cout << std::endl;

	//std::cout << "Timing stdext::hash_map<std::string, int>" << std::endl;
	//timemap<stdext::hash_map<key_type, int> >(n, i);
	//std::cout << std::endl;

	//std::cout << "Timing std::map<std::string, int>" << std::endl;
	//timemap<std::map<key_type, int> >(n, i);
	//std::cout << std::endl;
}