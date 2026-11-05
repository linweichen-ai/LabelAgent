"""Map complete original SAM polygons without raster-to-contour reconstruction."""
import copy,hashlib
import numpy as np
def map_complete_response(response,box,width,height,label='pallet',shape_id=1):
 from source.agent.edit_agent import _rasterize,valid_shape
 x1,y1,x2,y2=box
 if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('Invalid original source crop')
 local={'id':shape_id,'label':label,'parts':copy.deepcopy(response.get('parts',[]))}
 if not valid_shape(local,x2-x1,y2-y1):raise ValueError('Reject complete invalid SAM geometry; do not drop parts')
 parts=[]
 for part in local['parts']:
  mapped=copy.deepcopy(part);mapped['points']=[[x+x1,y+y1] for x,y in part['points']];mapped['holes']=[[[x+x1,y+y1] for x,y in hole] for hole in part.get('holes',[])];parts.append(mapped)
 shape={'id':shape_id,'label':label,'parts':parts,'points':parts[0]['points'],'holes':parts[0].get('holes',[]),'score':response.get('score')}
 if not valid_shape(shape,width,height):raise ValueError('Reject complete mapped SAM geometry')
 expected=np.zeros((height,width),np.uint8);expected[y1:y2,x1:x2]=_rasterize(local,x2-x1,y2-y1);actual=_rasterize(shape,width,height)
 if not np.array_equal(expected,actual):raise ValueError('Original wire geometry mapping changed pixels')
 return shape,actual,{'all_parts_and_holes_kept':True,'removed_pixels':0,'added_pixels':0,'raster_sha256':hashlib.sha256(actual.tobytes()).hexdigest()}
