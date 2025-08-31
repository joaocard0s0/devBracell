from shapely.validation import make_valid
from rasterio.mask import mask
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import rasterio
import zipfile
import shapely
import shutil
import cv2
import os

def read_polygons()->gpd.GeoDataFrame:
    name_file = "polygons_test.GeoJSON"
    # Reading file polygons
    gdf_polygons = gpd.read_file(os.path.join('GeoJson', name_file))

    return gdf_polygons

def read_uc()->gpd.GeoDataFrame:
    name_file = "layer_UCs.GeoJSON"
    # Reading file layer UC
    gdf_uc = gpd.read_file(os.path.join('GeoJson', name_file))
    # Fix any geometry invalid
    gdf_uc.geometry = list(map(lambda x: make_valid(x), gdf_uc.geometry))
    
    return gdf_uc

def explode_geometry(gdf:gpd.GeoDataFrame)->gpd.GeoDataFrame:
    """Explode geodataframe and reset index

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to explode

    Returns:
        gpd.GeoDataFrame: GeoDataFrame exploded
    """
    gdf = gdf.reset_index(drop=True)
    gdf = gdf.explode(ignore_index=True)

    return gdf

def search_and_copy_files(main_folder_path, destination_folder_path, file_name_parts):
    for root, dirs, files in os.walk(main_folder_path):
        for file in files:
            for file_name_part in file_name_parts:
                if file_name_part in file and not "MSK" in file:
                    file_path = os.path.join(root, file)
                    shutil.copy(file_path, destination_folder_path)
                    print(f'File {file} copied to {destination_folder_path}')

def check_files():
    """You need to download the files and leave them in the drive folder with keys name
    """
    base_path = os.path.join("Imagem", "drive")
    os.makedirs(base_path, exist_ok=True)

    links = {"T22LHH_20211006T132239_4326" :"https://drive.google.com/file/d/1sV3TGy9OLnvhzAcZF5Qo3laSb2UXDJUN/view?usp=sharing",
            "T22LHH_20211220T132234" :"https://drive.google.com/file/d/1vr_vnW-fZsoTyV5gkt7M7t73MlGzEcmC/view?usp=sharing",
            "T22LHH_20220228T132236" :"https://drive.google.com/file/d/1y7ki4K21T2xUQx0d3JiKBTzOrZxFh8OG/view?usp=sharing"}
   
    for folder , url in links.items():
        output = os.path.join(f"{os.path.join(base_path, folder)}")
        path_zip = f'{output}.zip'
        # Dowload files
        if not os.path.exists(path_zip):
           print(f"DO DOWLOAD {url}")
           print(f"Insert zip into {base_path}")
           quit()
        
        # Extract zip file
        if not os.path.isdir(output):
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(output) 

                # Get band 04 and 08 from link
                if folder == "T22LHH_20211220T132234" or folder == "T22LHH_20220228T132236":
                    search_and_copy_files(output, output, ["B04.jp2","B08.jp2"])


def save_tif(output_path:str, image:np.array, out_meta:dict):
    """Save tif

    Args:
        output_path (str): path to save tif
        image (np.array): array with image
        out_meta (dict): meta from src.meta
        type (rasterio.dtype, Optional): dtype to save file. Default float32
    """
    bands = 1 if len(image.shape) < 3 else image.shape[0]
    out_meta.update({"count": bands})
    
    if os.path.exists(output_path):return
    with rasterio.open(output_path, "w", **out_meta) as dest:
        if bands == 1 and len(image.shape) < 3:
            dest.write(image, 1)
        elif bands == 1 and len(image.shape) == 3:
            dest.write(image[0,:,:], 1)
        else:
            dest.write(image)

def calc_NDVI(array:np.array)->np.array:
    """Calc NDVI with array, 
       channel  | channel 0 = red    |  
                | channel -1 = NIR |     
               
    Args:
        array (np.array): array with R in first channel and NIR in last

    Returns:
        np.array: NDVI array
    """    
    # Get red band
    R = array[0, :, :].astype(rasterio.float32)
    # Get NIR band
    NIR = array[-1, :, :].astype(rasterio.float32)
    
    # Calc 
    num = (NIR - R)
    den = (NIR + R)
    # Division without 0
    ndvi = np.where(den == 0, 0, num / den)

    return ndvi

def clip_image(path_output_clip:str, path_image:str, gdf_polygons:gpd.GeoDataFrame):
    """Clip image tif by field 

    Args:
        path_output_clip (str): path to save file .tif
        path_image (str): path origin tif
        gdf_polygons (gpd.GeoDataFrame): field to clip
    """
    # Clip image 
    with rasterio.open(path_image) as src:
        crs_image = src.crs['init'].upper()
        if gdf_polygons.crs != crs_image:
            gdf_polygons = gdf_polygons.to_crs(crs_image)

        out_image_field, out_transform_field = mask(src, gdf_polygons.geometry, crop=True)
        
        # Update meta
        out_meta = src.profile.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image_field.shape[1],
            "width": out_image_field.shape[2],
            "transform": out_transform_field
        })
        # Save tif
        save_tif(path_output_clip, out_image_field, out_meta)

def read_tif(path:str)->tuple:
    """Read tif with rasterio

    Args:
        path (str): image path

    Returns:
        tuple: array and src
    """
    with rasterio.open(path) as src:
        array = src.read()

        return array, src

def test_01()->gpd.GeoDataFrame:
    # Read polygons
    gdf_polygons = read_polygons()
    # Find invalid geometry
    invalid_geometry = gdf_polygons[~gdf_polygons.geometry.is_valid]
    # Count geometry invalid
    print(f'Have {len(invalid_geometry)} invalid geometry')
    # Fix geometry
    gdf_polygons.geometry = list(map(lambda x: make_valid(x), gdf_polygons.geometry))

    return gdf_polygons

def test_02()->gpd.GeoDataFrame:
    # Read polygons
    gdf_polygons = test_01()
    # Calc area in hectare
    meters_per_hectare = 1e4
    # Find utm crs
    crs_utm =  gdf_polygons.estimate_utm_crs()
    # Calc area by polygon
    gdf_polygons['area_ha'] = gdf_polygons.to_crs(crs_utm).area/meters_per_hectare
    # Show area
    print(f'Have {round(gdf_polygons.area_ha.sum(), 2)} hectares in layer_UCs')

    return gdf_polygons

def test_03()->gpd.GeoDataFrame:
    # Reading geodataframe 
    gdf_polygons = test_01()
    gdf_uc = read_uc()

    #Multipart to single part
    gdf_polygons = explode_geometry(gdf_polygons)
    gdf_uc = explode_geometry(gdf_uc)

    # Columns to get informations
    gdf_polygons['intersect_area'] = 0.0
    gdf_polygons['UC_intersected'] = None

    # Informations
    meters_per_hectare = 1e4
    crs_utm = gdf_polygons.estimate_utm_crs()

    # For each polygon find intersection area with UC
    for i, polygon in gdf_polygons.iterrows():
        # Get just intersection area
        intersection = gdf_uc[gdf_uc.intersects(polygon.geometry)]
        # Sum area in hectare
        just_intersection = intersection.intersection(polygon.geometry)
        if len(just_intersection) > 0:
            area_ha = sum(just_intersection.to_crs(crs_utm).area/meters_per_hectare)
        else:
            area_ha = 0

        # Names UC intersected concated with ","
        name_uc_intesected = ', '.join(list(intersection.nome))
        # Set informations in polygons
        gdf_polygons.at[i, 'intersect_area'] = round(area_ha, 2)
        gdf_polygons.at[i, 'UC_intersected'] = name_uc_intesected
    
    return gdf_polygons

def test_04(numpy_ndvi:bool=False):
    """Test 04

    Args:
        numpy_ndvi (bool, optional): Flag to get np.array, NDVI. Defaults to False.

    Returns:
        str|(np.array, rasterio.src): path with NDVI or array and src
    """
    # Check files
    check_files()

    # Get image path
    path_base = os.path.join("Imagem", "output")
    os.makedirs(path_base, exist_ok=True)
    path_image = os.path.join("Imagem", "drive", "T22LHH_20211006T132239_4326", "T22LHH_20211006T132239_4326.tif")
    # Path to output cliped image
    path_output_clip = os.path.join(path_base, "cliped_by_fied.tif")
    # Path to NDVI
    path_output_NDVI = os.path.join(path_base, "NDVI.tif")

    # Check if file exists
    if os.path.exists(path_output_NDVI) and not numpy_ndvi:
        return path_output_NDVI
    
    # Reading file
    gdf_polygons = test_01()

    # CLip image
    clip_image(path_output_clip, path_image, gdf_polygons)

    # NDVI equation (NIR (4) - red (1)) / (NIR (4) + red (1))
    with rasterio.open(path_output_clip) as src:
        # Get meta
        meta = src.profile
        # read array
        shape = src.read()
        # Calc ndvi
        ndvi = calc_NDVI(shape)

        if numpy_ndvi:
            return ndvi, src
        
        # Update meta
        ndvi_meta = meta.copy()
        ndvi_meta.update(dtype=rasterio.float32)
        # Save image .tif
        save_tif(path_output_NDVI, ndvi, ndvi_meta)
    
    return path_output_NDVI

def test_05()->gpd.GeoDataFrame:
    # Reading file
    gdf_polygons = test_01()
    # Array NDVI
    ndvi, src = test_04(numpy_ndvi=True)

    # Calc stats NDVI (MIN, MAX, MEAN, STD) to each field
    stats = zonal_stats(gdf_polygons, ndvi, affine=src.transform, stats=['mean', 'min', 'max', 'std'])

    # Add values by field
    gdf_polygons['mean'] = [stat['mean'] for stat in stats]
    gdf_polygons['min'] = [stat['min'] for stat in stats]
    gdf_polygons['max'] = [stat['max'] for stat in stats]
    gdf_polygons['std'] = [stat['std'] for stat in stats]

    return gdf_polygons

def test_06():
    # Reading file
    gdf_polygons = test_01()
    # Path ndvi
    path_output_NDVI = test_04()
    # Path to save maps
    path_base_save = os.path.join("Imagem", "output", "Mapas")
    os.makedirs(path_base_save, exist_ok=True)

    # Map NDVI to each polygon
    with rasterio.open(path_output_NDVI) as src:

        for i, polygon in gdf_polygons.iterrows():
            image, transform = mask(src, [shapely.geometry.mapping(polygon.geometry)], crop=True)

            ndvi = image[0]
            cmap = plt.cm.Spectral
            norm = plt.Normalize(vmin=-1, vmax=1)  
            ndvi_colored = cmap(norm(ndvi))

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(ndvi_colored, extent=(transform[2], transform[2] + transform[0] * image.shape[2],
                                            transform[5] + transform[4] * image.shape[1], transform[5]))
            
            ax.set_title(f'NDVI - Talhão {i}')
            ax.axis('off')

            # Add geometry field
            gdf_polygons.loc[[i]].boundary.plot(ax=ax, edgecolor='black')

            # save map PDF
            out_pdf_path = os.path.join(path_base_save, f'talhao_{i}_ndvi.pdf')
            plt.savefig(out_pdf_path, bbox_inches='tight', dpi=300)
            plt.close()

def test_07():
    return test_05()

def test_08():
    # Sentinel-2
    # band 4 ==  RED
    # band 8 ==  NIR
    band_red = "B04"
    band_nir = "B08"

    # Reading file
    gdf_polygons = test_01()
    gdf_polygons = gdf_polygons.iloc[[1, 3]]

    # Check files
    check_files()

    # Need clip all images by field
    base_path_serie = os.path.join("Imagem", "drive")
    # Folder to clip images
    folders = ["T22LHH_20211220T132234", "T22LHH_20220228T132236", "T22LHH_20211006T132239_4326"]
    ext_image = (".jp2", ".tif")

    # Join band red and NIR
    # CLip all images
    paths_images_cliped = {}
    for folder in folders:
        paths = []
        for file in os.scandir(os.path.join(base_path_serie,folder)):
            # Check if is a image
            if file.name.endswith(ext_image) and not "xml" in file.name and not "_cliped" in file.name:
                # Clip image
                ext = file.name.split('.')[-1]
                path_image = file.path.replace(f'.{ext}', f"_cliped.{ext}")
                 # Add path to list 
                paths.append(path_image)
                if os.path.isfile(path_image):continue 
                clip_image(path_image, file.path, gdf_polygons)
               
        
        # Get all paths with cliped image
        paths_images_cliped[folder] = paths
    
    # Create var to array
    ndvi_06_10_21 = (None, "T22LHH_20211006T132239_4326")
    ndvi_20_12_21 = (None, "T22LHH_20211220T132234")
    ndvi_28_02_22 = (None, "T22LHH_20220228T132236")

    # Read array and calc NDVI
    for folder, path_file in paths_images_cliped.items():
        for file in path_file:
            if band_red in file:
                red, _ = read_tif(file)

            elif band_nir in file:
                nir, _ = read_tif(file)
            
            else:
                ndvi_06_10 = read_tif(file)[0]

        if folder == ndvi_06_10_21[1]:
            ndvi_06_10_21 = (calc_NDVI(ndvi_06_10), "T22LHH_20211006T132239_4326")
            continue

        # Concatenate values
        join_band = np.concatenate((red, nir))
        # Calc NDVI 
        ndvi = calc_NDVI(join_band)
        # assign to var
        if folder == ndvi_20_12_21[1]:
            ndvi_20_12_21 = (ndvi, folder)
        elif folder == ndvi_28_02_22[1]:
            ndvi_28_02_22 = (ndvi, folder)
    
    ndvi_images = [ (ndvi_06_10_21[0], "06/10/2021"),
                    (ndvi_20_12_21[0], "20/12/2021"),
                    (ndvi_28_02_22[0], "28/02/2022")
                    ]
    
    # Create pdf
    plt.figure(figsize=(12, 14)) 
    for i, (ndvi_array, date) in enumerate(ndvi_images):
        ax = plt.subplot(3, 1, i + 1)
        img = plt.imshow(ndvi_array, cmap='RdYlGn')
        plt.title(f"NDVI {date}", fontsize=16)
        plt.axis('off')

        cbar = plt.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.1, ax=ax)
        cbar.set_label('NDVI Value', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Salva o PDF
    plt.tight_layout()
    base_path = os.path.join("Imagem", "output")
    os.makedirs(base_path, exist_ok=True)
    plt.savefig(os.path.join(base_path, 'Serie_temporal_teste 08.pdf'))

def test_09():
    # Path image
    file_path = os.path.join("Imagem","Teste.jpg")
    # Path mask
    base_path = os.path.join("Imagem", "output")
    os.makedirs(base_path, exist_ok=True)
    mask_path = os.path.join(base_path, "mask_test_09.jpg")
    # Reading file
    with rasterio.open(file_path) as src:
        array = src.read()
        # Get bands
        red = array[0,:,:].astype(rasterio.uint8)
        green = array[1,:,:].astype(rasterio.uint8)
        blue = array[2,:,:].astype(rasterio.uint8)
         
        #Calc VARI
        den = green - red
        num = green + red - blue
        vari = np.where(den == 0, 0, num / den)

        # Limiar 
        min_th = 2
        max_th = 8
        vari = np.where((vari >= min_th) & (vari <= max_th), vari, 0)

        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        vari = cv2.dilate(vari, kernel, iterations=1)
        vari = cv2.erode(vari, kernel, iterations=1)
        
        meta = src.profile.copy()
        meta.update(dtype=rasterio.uint8)

        save_tif(mask_path, vari, meta)


if __name__ == "__main__":
    test_09()