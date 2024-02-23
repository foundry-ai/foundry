{{ ("``" + (fullname) + "`` module") | underline}}

{% if modules %}
.. toctree::
   :hidden:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

{% if functions %}
.. rubric:: Functions
.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Globals
.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes
.. autosummary::
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions
{% for item in functions %}
.. autofunction:: {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes
{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :special-members: __call__
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

{%- endfor %}
{% endif %}

.. The autosummary-related directives that are hidden in the output are used to
   generate the module index. They are hidden because they are not meant to be
   seen by the user, but are meant to be consumed by Sphinx:

{% if modules %}
.. rst-class:: hidden

   Hidden:

.. autosummary::
   :toctree:
      :hidden:
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}