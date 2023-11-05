ARG OPENSEARCH_VERSION=latest
FROM opensearchproject/opensearch:$OPENSEARCH_VERSION

# OPENSEARCH_VERSION needs to be redefined as any arg before FROM is outside build scope.
ARG OPENSEARCH_VERSION=latest
ARG opensearch_path=/usr/share/opensearch
ARG opensearch_yml=$opensearch_path/config/opensearch.yml

ARG SECURE_INTEGRATION
RUN echo "plugins.ml_commons.only_run_on_ml_node: false" >> $opensearch_yml;
RUN echo "plugins.ml_commons.native_memory_threshold: 100" >> $opensearch_yml;
RUN if [ "$OPENSEARCH_VERSION" == "2.11.0" ] ; then echo "plugins.ml_commons.model_access_control_enabled: true" >> $opensearch_yml; fi
RUN if [ "$SECURE_INTEGRATION" != "true" ] ; then echo "plugins.security.disabled: true" >> $opensearch_yml; fi
