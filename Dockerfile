# Use base AI image with Python dependencies.
FROM gadgetron_ai_base:0.1

# Copy Python code to correct place.
COPY --chown=vscode:conda code/*.py /opt/conda/envs/gadgetron/share/gadgetron/python/
COPY --chown=vscode:conda code/modules/*.py /opt/conda/envs/gadgetron/share/gadgetron/python/modules/
COPY --chown=vscode:conda code/modules/schemas/*.py /opt/conda/envs/gadgetron/share/gadgetron/python/modules/schemas/
# Copy parameters to the correct place.
COPY --chown=vscode:conda code/modules/parameters/*.pt /opt/conda/envs/gadgetron/share/gadgetron/python/
# Copy config file to the correct place.
COPY --chown=vscode:conda code/*.xml /opt/conda/envs/gadgetron/share/gadgetron/config/ 
