from enum import Enum, Flag


class S3L2SYNCloudFlags(Flag):
    """
    Flag meanings are described in the SYNERGY Product Data Format Specification [1]
    Table 23
    """

    CLOUD = 1
    CLOUD_AMBIGUOUS = 1 << 1
    CLOUD_MARGIN = 1 << 2
    SNOW_ICE = 1 << 3


class S3L2SYNClassificationAerosolFlags(Flag):
    """
    Flag meanings are described in the SYNERGY Product Data Format Specification [1]
    Talble 19


    [1] https://sentinels.copernicus.eu/documents/247904/1872824/S3IPF+PDS+006+-+i1r15+-+Product+Data+Format+Specification+-+SYNERGY_20221208.pdf/48f4eb8c-ca08-eca1-f8dd-bf00c438ea52?t=1683308328839
    """

    SYN_AOT_climato = 1 << 3
    SYN_land = 1 << 4
    SYN_no_olc = 1 << 5
    SYN_no_sln = 1
    # TODO ...


class S2L2ASCFlags(Enum):
    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    CAST_SHADOWS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11
