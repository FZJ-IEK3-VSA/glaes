import glaes as gl
import geokit as gk

class WindTurbineExclusionSets(object):
    
    @staticmethod
    def Holtinger2016_max(reg):
        """
        SOURCE: Holtinger2016

        EXCLUSIONS:
            - Areas above the alpine forest line
            - Maximum slope (11.3 degrees)
            - Water bodies
            - Settlement areas - 1000m buffer
            - Buildings outside of settlement areas - 750m buffer
            - Building land outside of settlement areas - 750m buffer
            - Built-up areas - 300m buffer
                * Industrial, commercial, and mining areas
            - Railways - 300m buffer
            - Motorways, primary, and secondary roads - 300m buffer
            - Airport public safety zones - 5100m buffer
            - Power grid( >110kV) - 250m buffer
            - National parks - 1000m buffer
            - Natura 2000 - habitats directive sites 
                * Potentially
            - Natura 2000 - birds directive sites
            - Other protected areas
                * Biosphere reserves, landscape protection areas, natural monuments and sites, 
            - Important birdlife areas
                * Potentially
            - Major migration routes for wild animals
                * Potentially
            - Lakes (> 50 ha) - 1000m buffer
        """
        ec = gl.ExclusionCalculator(reg)
        
        ##### do exclusions #####
        # "Areas above the alpine forest line
        ec.excludePrior("elevation_threshold", value=(1750, None) ) # alpine forest line assumed at 1750 m
        
        # "maximum slope (degrees)"
        ec.excludePrior("slope_threshold", value=(11.3, None) )
        
        # "water bodies"
        ec.excludePrior("river_proximity", value=0 )
        #ec.excludePrior("lake_proximity", value=0 ) # commented out since it is included late with a buffer
        
        # "settlement areas - 1000m buffer"
        ec.excludePrior("settlement_proximity", value=(None, 1000) )
        
        # "buildings outside of settlement areas - 750m buffer"
        
        # "building land outside of settlement areas - 750m buffer"
        
        # "built up areas - 300m buffer"
        # "  * industrial, commercial, and mining areas"
        ec.excludePrior("industrial_proximity", value=(None, 300) )
        ec.excludePrior("mining_proximity", value=(None, 300) )
        
        # "railways - 300m buffer"
        ec.excludePrior("railway_proximity", value=(None, 300) )
        
        # "motorways, primary, and secondary roads - 300m buffer"
        ec.excludePrior("roads_main_proximity", value=(None, 300) )
        ec.excludePrior("roads_secondary_proximity", value=(None, 300) )
        
        # "airport public safety zones - 5100m buffer"
        ec.excludePrior("airport_proximity", value=(None, 5100) )
        
        # "power grid( >110kV) - 250m buffer"
        ec.excludePrior("power_line_proximity", value=(None, 250) )
        
        # "national parks - 1000m buffer"
        ec.excludePrior("protected_park_proximity", value=(None,1000) )

        # "Natura 2000 - habitats directive sites" 
        # "*potentially"
        ec.excludePrior("protected_habitat_proximity", value=0 )

        # "Natura 2000 - birds directive sites"
        ec.excludePrior("protected_bird_proximity", value=0 )

        # "Other protected areas"
        # "*Biosphere reserves, landscape protection areas, natural monuments and sites, 
        #   protected habitats, and landscape section"
        ec.excludePrior("protected_biosphere_proximity", value=0 )
        ec.excludePrior("protected_landscape_proximity", value=0 )
        ec.excludePrior("protected_natural_monument_proximity", value=0 )

        # "important birdlife areas"
        # "*potentially"

        # "major migration routes for wild animals"
        # "*potentially"

        # "lakes (> 50 ha) - 1000m buffer"
        ec.excludePrior("lake_proximity", value=(None,1000) )
        
        # All done
        return ec

    @staticmethod
    def Holtinger2016_med(reg):
        """
        SOURCE: Holtinger2016

        EXCLUSIONS:
            - Areas above the alpine forest line
            - Maximum slope (8.5 degrees)
            - Water bodies
            - Settlement areas - 1200m buffer
            - Buildings outside of settlement areas - 750m buffer
            - Building land outside of settlement areas - 750m buffer
            - Built-up areas - 300m buffer
                * Industrial, commercial, and mining areas
            - Railways - 300m buffer
            - Motorways, primary, and secondary roads - 300m buffer
            - Airport public safety zones - 5100m buffer
            - Power grid( >110kV) - 250m buffer
            - National parks - 2000m buffer
            - Natura 2000 - habitats directive sites 
                * Potentially
            - Natura 2000 - birds directive sites
            - Other protected areas
                * Biosphere reserves, landscape protection areas, natural monuments and sites, 
            - Important birdlife areas
                * Potentially
            - Major migration routes for wild animals
                * Potentially
            - Forest areas
                * Only Exclude areas in communities with a forest share below 10%
            - Lakes (> 50 ha) - 1750m buffer
        """

        ec = gl.ExclusionCalculator(reg)
        
        ##### do exclusions #####
        # "Areas above the alpine forest line
        ec.excludePrior("elevation_threshold", value=(1750, None) ) # alpine forest line assumed at 1750 m
        
        # "maximum slope (8.5 degrees)"
        ec.excludePrior("slope_threshold", value=(8.5, None) )
        
        # "water bodies"
        ec.excludePrior("river_proximity", value=0 )
        #ec.excludePrior("lake_proximity", value=0 ) # commented out since it is included late with a buffer
        
        # "settlement areas - 1200m buffer"
        ec.excludePrior("settlement_proximity", value=(None, 1200) )
        
        # "buildings outside of settlement areas - 750m buffer"
        # maybe I should apply OSM directly, here
        
        # "building land outside of settlement areas - 750m buffer"
        # maybe I should apply OSM directly, here
        
        # "built up areas - 300m buffer"
        # "  * industrial, commercial, and mining areas"
        ec.excludePrior("industrial_proximity", value=(None, 300) )
        ec.excludePrior("mining_proximity", value=(None, 300) )
        
        # "railways - 300m buffer"
        ec.excludePrior("railway_proximity", value=(None, 300) )
        
        # "motorways, primary, and secondary roads - 300m buffer"
        ec.excludePrior("roads_main_proximity", value=(None, 300) )
        ec.excludePrior("roads_secondary_proximity", value=(None, 300) )
        
        # "airport public safety zones - 5100m buffer"
        ec.excludePrior("airport_proximity", value=(None, 5100) )
        
        # "power grid( >110kV) - 250m buffer"
        ec.excludePrior("power_line_proximity", value=(None, 250) )
        
        # "national parks - 2000m buffer"
        ec.excludePrior("protected_park_proximity", value=(None,2000) )

        # "Natura 2000 - habitats directive sites" 
        # "*potentially"
        ec.excludePrior("protected_habitat_proximity", value=0 )

        # "Natura 2000 - burds directive sites"
        ec.excludePrior("protected_bird_proximity", value=0 )

        # "Other protected areas"
        # "*Biosphere reserves, landscape protection areas, natural monuments and sites, 
        #   protected habitats, and landscape section"
        ec.excludePrior("protected_biosphere_proximity", value=0 )
        ec.excludePrior("protected_landscape_proximity", value=0 )
        ec.excludePrior("protected_natural_monument_proximity", value=0 )

        # "important birdlife areas"
        # "*potentially"

        # "major migration routes for wild animals"
        # "*potentially"

        # "forest areas"
        # "* excluding areas in communities with a forest share below 10%"

        # "lakes (> 50 ha) - 1750m buffer"
        ec.excludePrior("lake_proximity", value=(None,1750) )
        
        # All done
        return ec

    @staticmethod
    def Holtinger2016_min(reg):
        """
        SOURCE: Holtinger2016

        EXCLUSIONS:
            - Areas above the alpine forest line
            - Maximum slope (5.7 degrees)
            - Water bodies
            - Settlement areas - 2000m buffer
            - Buildings outside of settlement areas - 1000m buffer
            - Building land outside of settlement areas - 1000m buffer
            - Built up areas - 300m buffer
                * Industrial, commercial, and mining areas
            - Railways - 300m buffer
            - Motorways, primary, and secondary roads - 300m buffer
            - Airport public safety zones - 5100m buffer
            - Power grid( >110kV) - 250m buffer
            - National parks - 3000m buffer
            - Natura 2000 - habitats directive sites - 2000m buffer 
                * Potentially
            - Natura 2000 - burds directive sites - 2000m buffer
            - Other protected areas - 2000m buffer
                * Biosphere reserves, landscape protection areas, natural monuments and sites, 
            - Important birdlife areas
                * Potentially
            - Major migration routes for wild animals
                * Potentially
            - Forest areas - 1000m buffer
            - Lakes (> 50 ha) - 3000m buffer
        """
        ec = gl.ExclusionCalculator(reg)
        
        ##### do exclusions #####
        # "Areas above the alpine forest line
        ec.excludePrior("elevation_threshold", value=(1750, None) ) # alpine forest line assumed at 1750 m
        
        # "maximum slope (5.7 degrees)"
        ec.excludePrior("slope_threshold", value=(5.7, None) )
        
        # "water bodies"
        ec.excludePrior("river_proximity", value=0 )
        #ec.excludePrior("lake_proximity", value=0 ) # commented out since it is included late with a buffer
        
        # "settlement areas - 2000m buffer"
        ec.excludePrior("settlement_proximity", value=(None, 2000) )
        
        # "buildings outside of settlement areas - 1000m buffer"
        # maybe I should apply OSM directly, here
        
        # "building land outside of settlement areas - 1000m buffer"
        # maybe I should apply OSM directly, here
        
        # "built up areas - 300m buffer"
        # "  * industrial, commercial, and mining areas"
        ec.excludePrior("industrial_proximity", value=(None, 300) )
        ec.excludePrior("mining_proximity", value=(None, 300) )
        
        # "railways - 300m buffer"
        ec.excludePrior("railway_proximity", value=(None, 300) )
        
        # "motorways, primary, and secondary roads - 300m buffer"
        ec.excludePrior("roads_main_proximity", value=(None, 300) )
        ec.excludePrior("roads_secondary_proximity", value=(None, 300) )
        
        # "airport public safety zones - 5100m buffer"
        ec.excludePrior("airport_proximity", value=(None, 5100) )
        
        # "power grid( >110kV) - 250m buffer"
        ec.excludePrior("power_line_proximity", value=(None, 250) )
        
        # "national parks - 3000m buffer"
        ec.excludePrior("protected_park_proximity", value=(None,3000) )

        # "Natura 2000 - habitats directive sites - 2000m buffer" 
        # "*potentially"
        ec.excludePrior("protected_habitat_proximity", value=(None,2000) )

        # "Natura 2000 - burds directive sites - 2000m buffer"
        ec.excludePrior("protected_bird_proximity", value=(None,2000) )

        # "Other protected areas - 2000m buffer"
        # "*Biosphere reserves, landscape protection areas, natural monuments and sites, 
        #   protected habitats, and landscape section"
        ec.excludePrior("protected_biosphere_proximity", value=(None,2000) )
        ec.excludePrior("protected_landscape_proximity", value=(None,2000) )
        ec.excludePrior("protected_natural_monument_proximity", value=(None,2000) )

        # "important birdlife areas"
        # "*potentially"

        # "major migration routes for wild animals"
        # "*potentially"

        # "forest areas - 1000m buffer"
        ec.excludePrior("woodland_deciduous_proximity", value=(0,1000))
        ec.excludePrior("woodland_coniferous_proximity", value=(0,1000))
        ec.excludePrior("woodland_mixed_proximity", value=(0,1000))

        # "lakes (> 50 ha) - 30000m buffer"
        ec.excludePrior("lake_proximity", value=(None,3000) )
        
        # All done
        return ec
        
class ExclusionSets(object):
    Wind = WindTurbineExclusionSets()