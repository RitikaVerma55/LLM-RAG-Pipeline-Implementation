import csv
import json
import urllib.request
import argparse

from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='Scrap products data from the shopify store')
parser.add_argument('-t/--target', dest='website_url', type=str, help='URL to Shopify store (https://shopifystore.com)')
parser.add_argument('-v', '--variants', dest='variants',  action="store_true", help='Scrap also with variants data')
args = parser.parse_args()

if not args.website_url:
    print("usage: scraping.py [-h] [-t WEBSITE_URL] [-v]")
    exit(0)

#---------------------------------------------------------------------------------

base_url = args.website_url
url = base_url + '/products.json'
print(url)
with_variants = args.variants

def get_page(page):
    data = urllib.request.urlopen(url + '?page={}'.format(page)).read()
    products = json.loads(data)['products'] 

    return products

#----------------------------------------------------------------------------------
def get_tags_from_product(product):
    r = urllib.request.urlopen(product).read()
    soup = BeautifulSoup(r, "html.parser")

    title = soup.title.string
    description=''

    og_tags = soup.find_all('meta', attrs={"property": "og:description"})
    # print(meta)

    if og_tags:
          description = og_tags[0]['content']


    return [title, description]

#-----------------------------------------------------------------------------------------------------

def get_product_specs(product):
    specs={}

    page_html = urllib.request.urlopen(product).read()
    soup = BeautifulSoup(page_html, "html.parser")

    #Retrieving Size and measurements:
    sizes_list = soup.find('ul', class_='sizes-list')
    if sizes_list:
        size =[li.text.strip().replace('\n', ': ') for li in sizes_list.find_all('li')]
        specs['size']=size
        print(size)
    else:
        specs['size']= 'NA'


    #Retrieving material_list
    material_list = soup.find('ul', class_='materials-list')
    if material_list:
        material =[li.text.strip().replace('\n', ': ') for li in material_list.find_all('li')]
        specs['material']=material
        print(material)
    else:
        specs['material']= 'NA'


    #Retrieving compatibity:
    compatibility_list = soup.find('ul', class_='results-list')
    if compatibility_list:
        compatibility = [li.text.strip() for li in compatibility_list.find_all('li')]
        specs['compatibility']= compatibility
        print(compatibility)
    else:
        specs['compatibility']= 'NA'



    #Retrieving features:
    features_list = soup.find('ul', class_='features-list')
    if features_list:
        features = [li.text.strip() for li in features_list.find_all('li')]
        print(features)
        specs['features']= features
    else:
        specs['features']= 'NA'


    return specs

#----------------------------------------------------------------------------------------------------

with open('products.csv', 'w', encoding='utf-8', newline='') as f:
    page = 1

    print("[+] Starting script")

    # create file header
    writer = csv.writer(f)
    if with_variants:
        writer.writerow(['Name', 'Variant Name', 'Price', 'Size', 'Material', 'Compatibility', 'Features', 'URL', 'Meta Title', 'Meta Description', 'Product Description'])
    else:
        writer.writerow(['Name', 'URL', 'Meta Title', 'Meta Description', 'Product Description'])

    print("[+] Checking products page")

    products = get_page(page)
    while products:
        for product in products:
            name = product['title']
            product_url = base_url + '/products/' + product['handle']
            category = product['product_type']

            body_description = BeautifulSoup(product['body_html'], "html.parser")
            body_description = body_description.get_text()
          


            print(" ├ Scraping: " + product_url)

            title, description = get_tags_from_product(product_url)
            

            if with_variants:
                for variant in product['variants']:
                    variant_name = variant['title']
                    price = variant['price']
                    specs = get_product_specs(product_url)
                    Size = specs['size']
                    Material= specs['material']
                    Compatibility= specs['compatibility']
                    Features = specs['features']


                    row = [name, variant_name, price, Size, Material, Compatibility, Features, product_url, title, description, body_description]
                    writer.writerow(row)
            else:
                row = [name, product_url, title, description, body_description]
                writer.writerow(row)
        page += 1
        products = get_page(page)



 





