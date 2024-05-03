---
Title: Add A Table Of Contents to Your Articles
Tags: reST, markdown, navigation, web-design, contents
Category: Components
Date: 2014-03-18 11:03
Slug: how-elegant-displays-table-of-contents
Comment_id: lm95rjd-how-elegant-displays-table-of-contents
Summary: Elegant can be configured to provide a non-intrusive Table of Contents, giving a cleaner reading experience for the user.
Authors: Talha Mansoor, Jack De Winter
---

[TOC]

The key concept driving Elegant's design is to provide a reading experience that
is clean with minimal distractions. From that point of view, providing a table of
contents does not provide any additional information, but only exists to help guide the
reader through the article.

Therefore, Elegant places the table of contents on the left side
of the page with a relatively smaller font. This enables the table to guide the reader without
grabbing the focus of the reader and distracting them.

## Plugin Configuration

Enabling the Elegant' display of the table of contents is a two-step process.

In the first step,
use the markup languages to provide table of contents.

Second step is to take the generated table of contents and display it
on the left side of the article.

If both steps are not completed,
the table of contents will not be displayed on the left side of the article.

For the second step, you need to enable the `extract_toc` plugin in
your pelican configuration.

```python
PLUGINS = ['extract_toc']
```

## Configuring Markdown

You need to enable the `toc` extension for Markdown in your Pelican configuration.

```python
MARKDOWN = {
  'extension_configs': {
    'markdown.extensions.toc': {}
  }
}
```

Now to generate a table of contents for you article, add the `[TOC]` markdown tag to your
document.

```Markdown
Title: My sample title
Date: 2014-12-03
Category: Examples

[TOC]

## This is my first heading

This is the content of my sample blog post.

## This my second heading

I will end my example here.
```

### Other Options

For other options available for the Markdown Table of Contents extension, refer to the
[Python - Markdown - Table of Contents](https://python-markdown.github.io/extensions/toc/)
page.

### Debugging

1. Verify that your Markdown file has `[TOC]` tag
1. Verify that the
   `MARKDOWN` configuration variable is set properly.
1. Verify that the `PLUGINS` configuration variable is
   set properly.

## Configuring reStructuredText Format

reStructuredText format has the
[`contents` directive](http://docutils.sourceforge.net/docs/ref/rst/directives.html#table-of-contents)
that generates a table of contents in the article.

To generate a table of contents for you article, add the `.. contents::` directive to your document.

```rest
My sample title
###############

:date: 2014-12-03
:category: Examples

.. contents::

This is my first heading
========================

This is the content of my sample blog post.

This my second heading
======================

I will end my example here.
```

### Hide Default Title Text

!!! note "Possibly Deprecated"

    We couldn't not reproduce this issue in our testing. This should be considered deprecated, but is retained in this document in case someone encounters this.

Using the default configuration, reStructuredText will generate a default title for the table
of contents. According to the [official
documentation](http://docutils.sourceforge.net/docs/ref/rst/directives.html#table-of-contents),

> Language-dependent boilerplate text will be used for the title. The English
> default title text is "Contents".

This default configuration creates two titles for your table of contents, one generated by
reStructuredText and the other by Elegant. To disable default title generation
[^disable-title], you need to add following rule in your `custom.css` file to hide the
duplicate title:

```css
div#contents p.topic-title.first {
  display: none;
}
```

[^disable-title]: There is [no straightforward way](https://github.com/Pelican-Elegant/elegant/issues/54) to disable default title generation in reStructuredText.