#ifndef __SEGMENT_PARTS_HPP_INCLUDED__
#define __SEGMENT_PARTS_HPP_INCLUDED__


void segment_parts(object_info_t& oi,
		   vector<std::vector<std::string> > const &seq, int tt,
		   vector<CImg<unsigned char> > const& images,
		   vector<matrix<matrix<unsigned char> > >& seg_list);

void save_seg_list(std::string const& name,
		   vector<matrix<matrix<unsigned char> > > const& seg_list);

void load_seg_list(std::string const& name,
		   vector<matrix<matrix<unsigned char> > > & seg_list);

#endif
