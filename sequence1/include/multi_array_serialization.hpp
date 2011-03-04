//////////////////////////////////////////////////////////////////////////////
// detail::multi_array::serialization::serialize.hpp                        //
//                                                                          //
//  (C) Copyright 2009 Erwann Rogard                                        //
//  Use, modification and distribution are subject to the                   //
//  Boost Software License, Version 1.0. (See accompanying file             //
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        //
//////////////////////////////////////////////////////////////////////////////
#ifndef BOOST_STATISTICS_DETAIL_MULTI_ARRAY_SERIALIZATION_SERIALIZE_HPP_ER_2009
#define BOOST_STATISTICS_DETAIL_MULTI_ARRAY_SERIALIZATION_SERIALIZE_HPP_ER_2009

#include <boost/multi_array.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost{ namespace serialization{

	template<class Archive,class T>
	inline void load(
	    Archive & ar,
	    boost::multi_array<T,1> & t,
        const unsigned int file_version
	    )
	{
	    typedef boost::multi_array<T,1> multi_array_;
	    typedef typename multi_array_::size_type size_;
        
	    size_ n0;
	    ar >> BOOST_SERIALIZATION_NVP(n0);
            
	    t.resize(boost::extents[n0]);
	    ar >> make_array(t.data(), t.num_elements());
	}


	template<class Archive,class T>
	inline void load(
	    Archive & ar,
	    boost::multi_array<T,2> & t,
        const unsigned int file_version
	    )
	{
	    typedef boost::multi_array<T,2> multi_array_;
	    typedef typename multi_array_::size_type size_;
        
	    size_ n0;
	    ar >> BOOST_SERIALIZATION_NVP(n0);
	    size_ n1;
	    ar >> BOOST_SERIALIZATION_NVP(n1);
            
	    t.resize(boost::extents[n0][n1]);
	    ar >> make_array(t.data(), t.num_elements());
	}

	template<class Archive,class T>
	inline void load(
	    Archive & ar,
	    boost::multi_array<T,3> & t,
        const unsigned int file_version
	    )
	{
	    typedef boost::multi_array<T,3> multi_array_;
	    typedef typename multi_array_::size_type size_;
        
	    size_ n0;
	    ar >> BOOST_SERIALIZATION_NVP(n0);
	    size_ n1;
	    ar >> BOOST_SERIALIZATION_NVP(n1);
	    size_ n2;
	    ar >> BOOST_SERIALIZATION_NVP(n2);
            
	    t.resize(boost::extents[n0][n1][n2]);
	    ar >> make_array(t.data(), t.num_elements());
	}
    }}

namespace boost{ namespace serialization{

	template<typename Archive,typename T>
	inline void save(
	    Archive & ar,
	    const boost::multi_array<T,1> & t,
        const unsigned int file_version
	    )
	{
	    typedef boost::multi_array<T,1> multi_array_;
	    typedef typename multi_array_::size_type size_;
    
	    size_ n0 = (t.shape()[0]);
	    ar << BOOST_SERIALIZATION_NVP(n0);
	    ar << boost::serialization::make_array(t.data(), t.num_elements());
	}

	template<typename Archive,typename T>
	inline void save(
	    Archive & ar,
	    const boost::multi_array<T,2> & t,
	    const unsigned int file_version
	    )
	{
	    typedef boost::multi_array<T,2> multi_array_;
	    typedef typename multi_array_::size_type size_;
    
	    size_ n0 = (t.shape()[0]);
	    ar << BOOST_SERIALIZATION_NVP(n0);
	    size_ n1 = (t.shape()[1]);
	    ar << BOOST_SERIALIZATION_NVP(n1);
	    ar << boost::serialization::make_array(t.data(), t.num_elements());
	}

	template<typename Archive,typename T>
	inline void save(
	    Archive & ar,
	    const boost::multi_array<T,3> & t,
	    const unsigned int file_version
	    )
	{
	    typedef boost::multi_array<T,3> multi_array_;
	    typedef typename multi_array_::size_type size_;
    
	    size_ n0 = (t.shape()[0]);
	    ar << BOOST_SERIALIZATION_NVP(n0);
	    size_ n1 = (t.shape()[1]);
	    ar << BOOST_SERIALIZATION_NVP(n1);
	    size_ n2 = (t.shape()[2]);
	    ar << BOOST_SERIALIZATION_NVP(n2);
	    ar << boost::serialization::make_array(t.data(), t.num_elements());
	}

    } 
}

namespace boost{ namespace serialization{

	template<class Archive,typename T>
	inline void serialize(
	    Archive & ar,
	    boost::multi_array<T,1>& t,
        const unsigned int file_version
	    )
	{
	    split_free(ar, t, file_version);
	}

	template<class Archive,typename T>
	inline void serialize(
	    Archive & ar,
	    boost::multi_array<T,2>& t,
        const unsigned int file_version
	    )
	{
	    split_free(ar, t, file_version);
	}

	template<class Archive,typename T>
	inline void serialize(
	    Archive & ar,
	    boost::multi_array<T,3>& t,
        const unsigned int file_version
	    )
	{
	    split_free(ar, t, file_version);
	}

    }
}

#endif
