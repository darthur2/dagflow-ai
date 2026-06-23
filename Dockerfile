FROM rocker/r-ver:4.5.2

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('shiny', 'jsonlite', 'ggplot2', 'visNetwork', 'nleqslv', 'testthat'), repos='https://cloud.r-project.org', Ncpus=4)"

RUN npm install -g opencode-ai@1.15.13

WORKDIR /workspace
COPY . .

RUN cd .opencode && npm install

COPY opencode-config.json /etc/opencode-config.json
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
