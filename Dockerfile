FROM sevberg/geokit:latest
MAINTAINER sevberg "s.ryberg@fz-juelich.de"

# Install GLAES files
COPY setup.py MANIFEST.in LICENSE.txt README.md contributors.txt /repos/glaes/
COPY Examples /repos/glaes/Examples
COPY glaes /repos/glaes/glaes

# Install GLAES and test
RUN pip install -e /repos/glaes && \
    cd /repos/glaes/glaes/test && \
    python test_priors.py && \
    python test_ExclusionCalculator.py

# Setup entry
ENTRYPOINT ["/bin/bash"]
