import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Avatar,
  Badge,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Paper,
  Divider,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  CloudUpload,
  Lock,
  LockOpen,
  Search,
  Visibility,
  Delete,
  CheckCircle,
  Cancel,
  Person,
  LocationOn,
  Schedule,
  Security,
  History,
  Warning,
  Info
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const SuspectLock = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [suspects, setSuspects] = useState([]);
  const [selectedSuspect, setSelectedSuspect] = useState(null);
  const [matches, setMatches] = useState([]);
  const [auditLog, setAuditLog] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Create suspect dialog state
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newSuspect, setNewSuspect] = useState({
    name: '',
    description: '',
    priority: 'medium',
    tags: '',
    images: []
  });
  
  // Match verification dialog state
  const [verifyDialogOpen, setVerifyDialogOpen] = useState(false);
  const [selectedMatch, setSelectedMatch] = useState(null);
  
  // Search dialog state
  const [searchDialogOpen, setSearchDialogOpen] = useState(false);
  const [searchImage, setSearchImage] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [searchThreshold, setSearchThreshold] = useState(0.7);
  
  // Filters
  const [filters, setFilters] = useState({
    lockedOnly: false,
    priority: '',
    minConfidence: 0.5,
    verifiedOnly: false
  });

  // API base URL
  const API_BASE = '/api';

  // Load suspects on component mount
  useEffect(() => {
    loadSuspects();
  }, [filters.lockedOnly, filters.priority]);

  // Load matches when suspect is selected
  useEffect(() => {
    if (selectedSuspect) {
      loadMatches(selectedSuspect.target_id);
      loadAuditLog(selectedSuspect.target_id);
    }
  }, [selectedSuspect, filters.minConfidence, filters.verifiedOnly]);

  const loadSuspects = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filters.lockedOnly) params.append('locked_only', 'true');
      if (filters.priority) params.append('priority', filters.priority);
      
      const response = await fetch(`${API_BASE}/suspects?${params}`);
      const data = await response.json();
      
      if (response.ok) {
        setSuspects(data.targets);
      } else {
        setError(data.error || 'Failed to load suspects');
      }
    } catch (err) {
      setError('Network error loading suspects');
    } finally {
      setLoading(false);
    }
  };

  const loadMatches = async (targetId) => {
    try {
      const params = new URLSearchParams();
      params.append('limit', '50');
      if (filters.verifiedOnly) params.append('verified_only', 'true');
      if (filters.minConfidence > 0) params.append('min_confidence', filters.minConfidence.toString());
      
      const response = await fetch(`${API_BASE}/suspect/${targetId}/matches?${params}`);
      const data = await response.json();
      
      if (response.ok) {
        setMatches(data.matches);
      } else {
        setError(data.error || 'Failed to load matches');
      }
    } catch (err) {
      setError('Network error loading matches');
    }
  };

  const loadAuditLog = async (targetId) => {
    try {
      const response = await fetch(`${API_BASE}/audit/${targetId}?limit=20`);
      const data = await response.json();
      
      if (response.ok) {
        setAuditLog(data.audit_entries);
      } else {
        setError(data.error || 'Failed to load audit log');
      }
    } catch (err) {
      setError('Network error loading audit log');
    }
  };

  const createSuspect = async () => {
    try {
      setLoading(true);
      
      const formData = new FormData();
      formData.append('name', newSuspect.name);
      formData.append('description', newSuspect.description);
      formData.append('priority', newSuspect.priority);
      formData.append('tags', newSuspect.tags);
      formData.append('user_id', 'current_user'); // Replace with actual user ID
      
      newSuspect.images.forEach((image, index) => {
        formData.append('images', image);
      });
      
      const response = await fetch(`${API_BASE}/suspect`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSuccess('Suspect created successfully');
        setCreateDialogOpen(false);
        setNewSuspect({ name: '', description: '', priority: 'medium', tags: '', images: [] });
        loadSuspects();
      } else {
        setError(data.error || 'Failed to create suspect');
      }
    } catch (err) {
      setError('Network error creating suspect');
    } finally {
      setLoading(false);
    }
  };

  const toggleSuspectLock = async (targetId, locked) => {
    try {
      const response = await fetch(`${API_BASE}/suspect/${targetId}/lock`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          locked: locked,
          user_id: 'current_user' // Replace with actual user ID
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSuccess(`Suspect ${locked ? 'locked' : 'unlocked'} successfully`);
        loadSuspects();
        if (selectedSuspect && selectedSuspect.target_id === targetId) {
          setSelectedSuspect({ ...selectedSuspect, is_locked: locked });
        }
      } else {
        setError(data.error || 'Failed to update lock status');
      }
    } catch (err) {
      setError('Network error updating lock status');
    }
  };

  const deleteSuspect = async (targetId) => {
    if (!window.confirm('Are you sure you want to delete this suspect? This action cannot be undone.')) {
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE}/suspect/${targetId}?user_id=current_user`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSuccess('Suspect deleted successfully');
        loadSuspects();
        if (selectedSuspect && selectedSuspect.target_id === targetId) {
          setSelectedSuspect(null);
        }
      } else {
        setError(data.error || 'Failed to delete suspect');
      }
    } catch (err) {
      setError('Network error deleting suspect');
    }
  };

  const verifyMatch = async (targetId, matchId, verified) => {
    try {
      const response = await fetch(`${API_BASE}/suspect/${targetId}/verify/${matchId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          verified: verified,
          user_id: 'current_user' // Replace with actual user ID
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSuccess(`Match ${verified ? 'verified' : 'rejected'} successfully`);
        setVerifyDialogOpen(false);
        loadMatches(targetId);
        loadAuditLog(targetId);
      } else {
        setError(data.error || 'Failed to verify match');
      }
    } catch (err) {
      setError('Network error verifying match');
    }
  };

  const searchSuspects = async () => {
    if (!searchImage) {
      setError('Please select an image to search');
      return;
    }
    
    try {
      setLoading(true);
      
      const formData = new FormData();
      formData.append('image', searchImage);
      formData.append('threshold', searchThreshold.toString());
      
      const response = await fetch(`${API_BASE}/suspect/search`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setSearchResults(data.matches);
      } else {
        setError(data.error || 'Failed to search suspects');
      }
    } catch (err) {
      setError('Network error searching suspects');
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (event, type) => {
    const files = Array.from(event.target.files);
    
    if (type === 'create') {
      setNewSuspect({ ...newSuspect, images: [...newSuspect.images, ...files] });
    } else if (type === 'search') {
      setSearchImage(files[0]);
    }
  };

  const removeImage = (index) => {
    const newImages = newSuspect.images.filter((_, i) => i !== index);
    setNewSuspect({ ...newSuspect, images: newImages });
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'default';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          <Security sx={{ mr: 1, verticalAlign: 'middle' }} />
          Suspect Lock System
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<Search />}
            onClick={() => setSearchDialogOpen(true)}
            sx={{ mr: 1 }}
          >
            Search by Image
          </Button>
          <Button
            variant="contained"
            startIcon={<Person />}
            onClick={() => setCreateDialogOpen(true)}
          >
            Add Suspect
          </Button>
        </Box>
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Suspects List */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Suspects ({suspects.length})
              </Typography>
              
              {/* Filters */}
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={filters.lockedOnly}
                      onChange={(e) => setFilters({ ...filters, lockedOnly: e.target.checked })}
                    />
                  }
                  label="Locked Only"
                />
                <FormControl size="small" sx={{ ml: 2, minWidth: 120 }}>
                  <InputLabel>Priority</InputLabel>
                  <Select
                    value={filters.priority}
                    label="Priority"
                    onChange={(e) => setFilters({ ...filters, priority: e.target.value })}
                  >
                    <MenuItem value="">All</MenuItem>
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                    <MenuItem value="critical">Critical</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <List>
                  {suspects.map((suspect) => (
                    <ListItem
                      key={suspect.target_id}
                      button
                      selected={selectedSuspect?.target_id === suspect.target_id}
                      onClick={() => setSelectedSuspect(suspect)}
                    >
                      <Avatar sx={{ mr: 2 }}>
                        {suspect.is_locked ? <Lock /> : <Person />}
                      </Avatar>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {suspect.name || 'Unnamed Suspect'}
                            <Chip
                              label={suspect.priority}
                              color={getPriorityColor(suspect.priority)}
                              size="small"
                              sx={{ ml: 1 }}
                            />
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="caption" display="block">
                              Created: {formatTimestamp(suspect.created_timestamp)}
                            </Typography>
                            <Typography variant="caption" display="block">
                              Images: {suspect.reference_images_count}
                            </Typography>
                            {suspect.tags && suspect.tags.length > 0 && (
                              <Box sx={{ mt: 0.5 }}>
                                {suspect.tags.map((tag, index) => (
                                  <Chip
                                    key={index}
                                    label={tag}
                                    size="small"
                                    variant="outlined"
                                    sx={{ mr: 0.5, mb: 0.5 }}
                                  />
                                ))}
                              </Box>
                            )}
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleSuspectLock(suspect.target_id, !suspect.is_locked);
                          }}
                          color={suspect.is_locked ? 'error' : 'success'}
                        >
                          {suspect.is_locked ? <Lock /> : <LockOpen />}
                        </IconButton>
                        <IconButton
                          edge="end"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteSuspect(suspect.target_id);
                          }}
                          color="error"
                        >
                          <Delete />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Suspect Details */}
        <Grid item xs={12} md={8}>
          {selectedSuspect ? (
            <Card>
              <CardContent>
                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                  <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
                    <Tab label="Matches" />
                    <Tab label="Details" />
                    <Tab label="Audit Log" />
                  </Tabs>
                </Box>

                {/* Matches Tab */}
                {activeTab === 0 && (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6">
                        Live Matches ({matches.length})
                      </Typography>
                      <Box>
                        <TextField
                          label="Min Confidence"
                          type="number"
                          size="small"
                          value={filters.minConfidence}
                          onChange={(e) => setFilters({ ...filters, minConfidence: parseFloat(e.target.value) })}
                          inputProps={{ min: 0, max: 1, step: 0.1 }}
                          sx={{ width: 150, mr: 1 }}
                        />
                        <FormControlLabel
                          control={
                            <Switch
                              checked={filters.verifiedOnly}
                              onChange={(e) => setFilters({ ...filters, verifiedOnly: e.target.checked })}
                            />
                          }
                          label="Verified Only"
                        />
                      </Box>
                    </Box>
                    
                    <List>
                      {matches.map((match) => (
                        <ListItem key={match.match_id}>
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Chip
                                  label={`${(match.confidence * 100).toFixed(1)}%`}
                                  color={getConfidenceColor(match.confidence)}
                                  size="small"
                                  sx={{ mr: 1 }}
                                />
                                {match.verified && (
                                  <CheckCircle color="success" sx={{ mr: 1 }} />
                                )}
                                <Typography variant="body2">
                                  {formatTimestamp(match.timestamp)}
                                </Typography>
                              </Box>
                            }
                            secondary={
                              <Box>
                                {match.location && (
                                  <Typography variant="caption" display="block">
                                    <LocationOn sx={{ fontSize: 14, mr: 0.5 }} />
                                    Lat: {match.location.lat?.toFixed(6)}, 
                                    Lon: {match.location.lon?.toFixed(6)}, 
                                    Alt: {match.location.alt?.toFixed(1)}m
                                  </Typography>
                                )}
                                {match.camera_id && (
                                  <Typography variant="caption" display="block">
                                    Camera: {match.camera_id}
                                  </Typography>
                                )}
                                {match.verified_by && (
                                  <Typography variant="caption" display="block">
                                    Verified by: {match.verified_by} at {formatTimestamp(match.verification_timestamp)}
                                  </Typography>
                                )}
                              </Box>
                            }
                          />
                          <ListItemSecondaryAction>
                            <Button
                              size="small"
                              onClick={() => {
                                setSelectedMatch(match);
                                setVerifyDialogOpen(true);
                              }}
                              disabled={match.verified}
                            >
                              {match.verified ? 'Verified' : 'Verify'}
                            </Button>
                          </ListItemSecondaryAction>
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}

                {/* Details Tab */}
                {activeTab === 1 && (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Suspect Details
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2">Name:</Typography>
                        <Typography variant="body2" gutterBottom>
                          {selectedSuspect.name || 'Unnamed'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2">Priority:</Typography>
                        <Chip
                          label={selectedSuspect.priority}
                          color={getPriorityColor(selectedSuspect.priority)}
                          size="small"
                        />
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="subtitle2">Description:</Typography>
                        <Typography variant="body2" gutterBottom>
                          {selectedSuspect.description || 'No description'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2">Status:</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {selectedSuspect.is_locked ? (
                            <><Lock color="error" sx={{ mr: 1 }} />Locked</>
                          ) : (
                            <><LockOpen color="success" sx={{ mr: 1 }} />Unlocked</>
                          )}
                        </Box>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle2">Created:</Typography>
                        <Typography variant="body2">
                          {formatTimestamp(selectedSuspect.created_timestamp)}
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="subtitle2">Tags:</Typography>
                        <Box sx={{ mt: 1 }}>
                          {selectedSuspect.tags && selectedSuspect.tags.length > 0 ? (
                            selectedSuspect.tags.map((tag, index) => (
                              <Chip
                                key={index}
                                label={tag}
                                size="small"
                                variant="outlined"
                                sx={{ mr: 0.5, mb: 0.5 }}
                              />
                            ))
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              No tags
                            </Typography>
                          )}
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>
                )}

                {/* Audit Log Tab */}
                {activeTab === 2 && (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Audit Log ({auditLog.length})
                    </Typography>
                    <List>
                      {auditLog.map((entry) => (
                        <ListItem key={entry.entry_id}>
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <History sx={{ mr: 1, fontSize: 16 }} />
                                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                  {entry.action.toUpperCase()}
                                </Typography>
                                <Typography variant="caption" sx={{ ml: 1 }}>
                                  {formatTimestamp(entry.timestamp)}
                                </Typography>
                              </Box>
                            }
                            secondary={
                              <Box>
                                {entry.user_id && (
                                  <Typography variant="caption" display="block">
                                    User: {entry.user_id}
                                  </Typography>
                                )}
                                <Typography variant="caption" display="block">
                                  Details: {JSON.stringify(entry.details)}
                                </Typography>
                              </Box>
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Person sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary">
                    Select a suspect to view details
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      {/* Create Suspect Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Add New Suspect</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Name"
                value={newSuspect.name}
                onChange={(e) => setNewSuspect({ ...newSuspect, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={newSuspect.priority}
                  label="Priority"
                  onChange={(e) => setNewSuspect({ ...newSuspect, priority: e.target.value })}
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="critical">Critical</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                multiline
                rows={3}
                value={newSuspect.description}
                onChange={(e) => setNewSuspect({ ...newSuspect, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Tags (comma-separated)"
                value={newSuspect.tags}
                onChange={(e) => setNewSuspect({ ...newSuspect, tags: e.target.value })}
                helperText="Enter tags separated by commas"
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                component="label"
                variant="outlined"
                startIcon={<CloudUpload />}
                sx={{ mb: 2 }}
              >
                Upload Reference Images
                <VisuallyHiddenInput
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e, 'create')}
                />
              </Button>
              {newSuspect.images.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Selected Images ({newSuspect.images.length}):
                  </Typography>
                  {newSuspect.images.map((image, index) => (
                    <Chip
                      key={index}
                      label={image.name}
                      onDelete={() => removeImage(index)}
                      sx={{ mr: 1, mb: 1 }}
                    />
                  ))}
                </Box>
              )}
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={createSuspect}
            variant="contained"
            disabled={!newSuspect.name || newSuspect.images.length === 0 || loading}
          >
            {loading ? <CircularProgress size={20} /> : 'Create Suspect'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Match Verification Dialog */}
      <Dialog open={verifyDialogOpen} onClose={() => setVerifyDialogOpen(false)}>
        <DialogTitle>Verify Match</DialogTitle>
        <DialogContent>
          {selectedMatch && (
            <Box>
              <Typography variant="body1" gutterBottom>
                Confidence: {(selectedMatch.confidence * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" gutterBottom>
                Timestamp: {formatTimestamp(selectedMatch.timestamp)}
              </Typography>
              {selectedMatch.location && (
                <Typography variant="body2" gutterBottom>
                  Location: {selectedMatch.location.lat?.toFixed(6)}, {selectedMatch.location.lon?.toFixed(6)}
                </Typography>
              )}
              <Typography variant="body2" color="text.secondary">
                Please verify if this match is correct.
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              if (selectedMatch && selectedSuspect) {
                verifyMatch(selectedSuspect.target_id, selectedMatch.match_id, false);
              }
            }}
            color="error"
            startIcon={<Cancel />}
          >
            Reject
          </Button>
          <Button
            onClick={() => {
              if (selectedMatch && selectedSuspect) {
                verifyMatch(selectedSuspect.target_id, selectedMatch.match_id, true);
              }
            }}
            color="success"
            variant="contained"
            startIcon={<CheckCircle />}
          >
            Verify
          </Button>
        </DialogActions>
      </Dialog>

      {/* Search Dialog */}
      <Dialog open={searchDialogOpen} onClose={() => setSearchDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Search Suspects by Image</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <Button
                component="label"
                variant="outlined"
                startIcon={<CloudUpload />}
                fullWidth
              >
                Upload Search Image
                <VisuallyHiddenInput
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e, 'search')}
                />
              </Button>
              {searchImage && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Selected: {searchImage.name}
                </Typography>
              )}
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Similarity Threshold"
                type="number"
                value={searchThreshold}
                onChange={(e) => setSearchThreshold(parseFloat(e.target.value))}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                helperText="Minimum similarity score (0.0 - 1.0)"
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                onClick={searchSuspects}
                variant="contained"
                disabled={!searchImage || loading}
                fullWidth
              >
                {loading ? <CircularProgress size={20} /> : 'Search'}
              </Button>
            </Grid>
            {searchResults.length > 0 && (
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Search Results ({searchResults.length})
                </Typography>
                <List>
                  {searchResults.map((result) => (
                    <ListItem key={result.target_id}>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {result.name || 'Unnamed Suspect'}
                            <Chip
                              label={`${(result.similarity * 100).toFixed(1)}%`}
                              color={getConfidenceColor(result.similarity)}
                              size="small"
                              sx={{ ml: 1 }}
                            />
                            <Chip
                              label={result.priority}
                              color={getPriorityColor(result.priority)}
                              size="small"
                              sx={{ ml: 1 }}
                            />
                          </Box>
                        }
                        secondary={`Status: ${result.is_locked ? 'Locked' : 'Unlocked'}`}
                      />
                      <ListItemSecondaryAction>
                        <Button
                          size="small"
                          onClick={() => {
                            const suspect = suspects.find(s => s.target_id === result.target_id);
                            if (suspect) {
                              setSelectedSuspect(suspect);
                              setSearchDialogOpen(false);
                            }
                          }}
                        >
                          View
                        </Button>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSearchDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SuspectLock;